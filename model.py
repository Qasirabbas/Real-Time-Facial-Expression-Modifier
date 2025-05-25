import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
import time
import subprocess
import datetime
from PIL.PngImagePlugin import PngInfo
import re
from videohelpersuite.logger import logger
import json
import copy
import dill
import yaml
from ultralytics import YOLO
import checkpoint_pickle
from string import Template
current_directory = ''
models_dir = "./models"
os.makedirs(models_dir, exist_ok=True)
import folder_paths
import math
from comfy_utils import ProgressBar
import struct
import safetensors.torch
import logging
import itertools
# from videohelpersuite.nodes import NODE_CLASS_MAPPINGS
def load_torch_file(ckpt, safe_load=False, device=None):
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                logging.warning("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=checkpoint_pickle)
        if "global_step" in pl_sd:
            logging.debug(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd


from LivePortrait.live_portrait_wrapper import LivePortraitWrapper
from LivePortrait.utils.camera import get_rotation_matrix
from LivePortrait.config.inference_config import InferenceConfig

from LivePortrait.modules.spade_generator import SPADEDecoder
from LivePortrait.modules.warping_network import WarpingNetwork
from LivePortrait.modules.motion_extractor import MotionExtractor
from LivePortrait.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from LivePortrait.modules.stitching_retargeting_network import StitchingRetargetingNetwork
from collections import OrderedDict
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
cur_device = None
def get_device():
    global cur_device
    if cur_device == None:
        if torch.cuda.is_available():
            cur_device = torch.device('cuda:1')
            print("Uses CUDA device.")
        elif torch.backends.mps.is_available():
            cur_device = torch.device('cpu')
            print("Uses MPS device.")
        else:
            cur_device = torch.device('cpu')
            print("Uses CPU device.")
    return cur_device

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
def rgb_crop(rgb, region):
    return rgb[region[1]:region[3], region[0]:region[2]]

def rgb_crop_batch(rgbs, region):
    return rgbs[:, region[1]:region[3], region[0]:region[2]]
def get_rgb_size(rgb):
    return rgb.shape[1], rgb.shape[0]
def create_transform_matrix(x, y, s_x, s_y):
    return np.float32([[s_x, 0, x], [0, s_y, y]])

def get_model_dir(m):
    return os.path.join(models_dir, m)

def calc_crop_limit(center, img_size, crop_size):
    pos = center - crop_size / 2
    if pos < 0:
        crop_size += pos * 2
        pos = 0

    pos2 = pos + crop_size

    if img_size < pos2:
        crop_size -= (pos2 - img_size) * 2
        pos2 = img_size
        pos = pos2 - crop_size

    return pos, pos2, crop_size

def retargeting(delta_out, driving_exp, factor, idxes):
    for idx in idxes:
        #delta_out[0, idx] -= src_exp[0, idx] * factor
        delta_out[0, idx] += driving_exp[0, idx] * factor

class PreparedSrcImg:
    def __init__(self, src_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori):
        self.src_rgb = src_rgb
        self.crop_trans_m = crop_trans_m
        self.x_s_info = x_s_info
        self.f_s_user = f_s_user
        self.x_s_user = x_s_user
        self.mask_ori = mask_ori

import requests
from tqdm import tqdm
def get_format_widget_defaults(format_name):
    video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    results = {}
    for w in gen_format_widgets(video_format):
        if len(w[0]) > 2 and 'default' in w[0][2]:
            default = w[0][2]['default']
        else:
            if type(w[0][1]) is list:
                default = w[0][1][0]
            else:
                #NOTE: This doesn't respect max/min, but should be good enough as a fallback to a fallback to a fallback
                default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[0][1]]
        results[w[0][0]] = default
    return results
def apply_format_widgets(format_name, kwargs):
    video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    for w in gen_format_widgets(video_format):
        assert(w[0][0] in kwargs)
        if len(w[0]) > 3:
            w[0] = Template(w[0][3]).substitute(val=kwargs[w[0][0]])
        else:
            w[0] = str(kwargs[w[0][0]])
    return video_format
def to_pingpong(inp):
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp)-2,0,-1):
        yield inp[i]
def tensor_to_int(tensor, bits):
    #TODO: investigate benefit of rounding by adding 0.5 before clip/cast
    tensor = tensor.cpu().numpy() * (2**bits-1)
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)
def gen_format_widgets(video_format):
    for k in video_format:
        if k.endswith("_pass"):
            for i in range(len(video_format[k])):
                if isinstance(video_format[k][i], list):
                    item = [video_format[k][i]]
                    yield item
                    video_format[k][i] = item[0]
        else:
            if isinstance(video_format[k], list):
                item = [video_format[k]]
                yield item
                video_format[k] = item[0]
def get_video_formats():
    formats = []
    for format_name in folder_paths.get_filename_list("VHS_video_formats"):
        format_name = format_name[:-5]
        video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name + ".json")
        with open(video_format_path, 'r') as stream:
            video_format = json.load(stream)
        if "gifski_pass" in video_format and gifski_path is None:
            #Skip format
            continue
        widgets = [w[0] for w in gen_format_widgets(video_format)]
        if (len(widgets) > 0):
            formats.append(["video/" + format_name, widgets])
        else:
            formats.append("video/" + format_name)
    return formats
class VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats = get_video_formats()
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (
                    "FLOAT",
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats,),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        frame_rate: int,
        loop_count: int,
        images=None,
        latents=None,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None,
        vae=None
    ):
        if latents is not None:
            images = latents
        if images is None:
            return ((save_output, []),)
        if vae is not None:
            if isinstance(images, dict):
                images = images['samples']
            else:
                vae = None

        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ((save_output, []),)
        num_frames = len(images)
        pbar = ProgressBar(num_frames)
        if vae is not None:
            downscale_ratio = getattr(vae, "downscale_ratio", 8)
            width = images.size(3)*downscale_ratio
            height = images.size(2)*downscale_ratio
            frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
            #Python 3.12 adds an itertools.batched, but it's easily replicated for legacy support
            def batched(it, n):
                while batch := tuple(itertools.islice(it, n)):
                    yield batch
            def batched_encode(images, vae, frames_per_batch):
                for batch in batched(iter(images), frames_per_batch):
                    image_batch = torch.from_numpy(np.array(batch))
                    yield from vae.decode(image_batch)
            images = batched_encode(images, vae, frames_per_batch)
            first_image = next(images)
            #repush first_image
            images = itertools.chain([first_image], images)
        else:
            first_image = images[0]
            images = iter(images)
        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files = []

        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        if meta_batch is not None and unique_id in meta_batch.outputs:
            (counter, output_process) = meta_batch.outputs[unique_id]
        else:
            # comfy counter workaround
            max_counter = 0

            # Loop through the existing files
            matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
            for existing_file in os.listdir(full_output_folder):
                # Check if the file matches the expected format
                match = matcher.fullmatch(existing_file)
                if match:
                    # Extract the numeric portion of the filename
                    file_counter = int(match.group(1))
                    # Update the maximum counter value if necessary
                    if file_counter > max_counter:
                        max_counter = file_counter

            # Increment the counter by 1 to get the next available value
            counter = max_counter + 1
            output_process = None

        # save first frame as png to keep metadata
        file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, file)
        print("First Image:", first_image)
        Image.fromarray(tensor_to_bytes(first_image)).save(
            file_path,
            pnginfo=metadata,
            compress_level=4,
        )
        output_files.append(file_path)

        format_type, format_ext = format.split("/")
        if format_type == "image":
            if meta_batch is not None:
                raise Exception("Pillow('image/') formats are not compatible with batched output")
            image_kwargs = {}
            if format_ext == "gif":
                image_kwargs['disposal'] = 2
            if format_ext == "webp":
                #Save timestamp information
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs['exif'] = exif
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            if pingpong:
                images = to_pingpong(images)
            frames = map(lambda x : Image.fromarray(tensor_to_bytes(x)), images)
            # Use pillow directly to save an animated image
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs
            )
            output_files.append(file_path)
        else:
            # Use ffmpeg to save a video
            if ffmpeg_path is None:
                raise ProcessLookupError(f"ffmpeg is required for video outputs and could not be found.\nIn order to use video outputs, you must either:\n- Install imageio-ffmpeg with pip,\n- Place a ffmpeg executable in {os.path.abspath('')}, or\n- Install ffmpeg and add it to the system path.")

            #Acquire additional format_widget values
            kwargs = None
            if manual_format_widgets is None:
                if prompt is not None:
                    kwargs = prompt[unique_id]['inputs']
                else:
                    manual_format_widgets = {}
            if kwargs is None:
                kwargs = get_format_widget_defaults(format_ext)
                missing = {}
                for k in kwargs.keys():
                    if k in manual_format_widgets:
                        kwargs[k] = manual_format_widgets[k]
                    else:
                        missing[k] = kwargs[k]
                if len(missing) > 0:
                    logger.warn("Extra format values were not provided, the following defaults will be used: " + str(kwargs) + "\nThis is likely due to usage of ComfyUI-to-python. These values can be manually set by supplying a manual_format_widgets argument")

            video_format = apply_format_widgets(format_ext, kwargs)
            has_alpha = first_image.shape[-1] == 4
            dim_alignment = video_format.get("dim_alignment", 8)
            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                #output frames must be padded
                to_pad = (-first_image.shape[1] % dim_alignment,
                          -first_image.shape[0] % dim_alignment)
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                           to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)
                def pad(image):
                    image = image.permute((2,0,1))#HWC to CHW
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1,2,0))
                images = map(pad, images)
                new_dims = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                            -first_image.shape[0] % dim_alignment + first_image.shape[0])
                dimensions = f"{new_dims[0]}x{new_dims[1]}"
                logger.warn("Output images were not of valid resolution and have had padding applied")
            else:
                dimensions = f"{first_image.shape[1]}x{first_image.shape[0]}"
            if loop_count > 0:
                loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(num_frames)]
            else:
                loop_args = []
            if pingpong:
                if meta_batch is not None:
                    logger.error("pingpong is incompatible with batched output")
                images = to_pingpong(images)
            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = map(tensor_to_shorts, images)
                if has_alpha:
                    i_pix_fmt = 'rgba64'
                else:
                    i_pix_fmt = 'rgb48'
            else:
                images = map(tensor_to_bytes, images)
                if has_alpha:
                    i_pix_fmt = 'rgba'
                else:
                    i_pix_fmt = 'rgb24'
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    "-s", dimensions, "-r", str(frame_rate), "-i", "-"] \
                    + loop_args

            images = map(lambda x: x.tobytes(), images)
            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])

            if "pre_pass" in video_format:
                if meta_batch is not None:
                    #Performing a prepass requires keeping access to all frames.
                    #Potential solutions include keeping just output frames in
                    #memory or using 3 passes with intermediate file, but
                    #very long gifs probably shouldn't be encouraged
                    raise Exception("Formats which require a pre_pass are incompatible with Batch Manager.")
                images = [b''.join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                pre_pass_args = args[:13] + video_format['pre_pass']
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env,
                                   capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n" \
                            + e.stderr.decode("utf-8"))
            if "inputs_main_pass" in video_format:
                args = args[:13] + video_format['inputs_main_pass'] + args[13:]

            if output_process is None:
                if 'gifski_pass' in video_format:
                    output_process = gifski_process(args, video_format, file_path, env)
                else:
                    args += video_format['main_pass'] + bitrate_arg
                    output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
                #Proceed to first yield
                output_process.send(None)
                if meta_batch is not None:
                    meta_batch.outputs[unique_id] = (counter, output_process)

            for image in images:
                pbar.update(1)
                output_process.send(image)
            if meta_batch is not None:
                requeue_workflow((meta_batch.unique_id, not meta_batch.has_closed_inputs))
            if meta_batch is None or meta_batch.has_closed_inputs:
                #Close pipe and wait for termination.
                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    pass
                if meta_batch is not None:
                    meta_batch.outputs.pop(unique_id)
                    if len(meta_batch.outputs) == 0:
                        meta_batch.reset()
            else:
                #batch is unfinished
                #TODO: Check if empty output breaks other custom nodes
                return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

            output_files.append(file_path)


            a_waveform = None
            if audio is not None:
                try:
                    #safely check if audio produced by VHS_LoadVideo actually exists
                    a_waveform = audio['waveform']
                except:
                    pass
            if a_waveform is not None:
                # Create audio file if input was provided
                output_file_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
                output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
                if "audio_pass" not in video_format:
                    logger.warn("Selected video format does not have explicit audio support")
                    video_format["audio_pass"] = ["-c:a", "libopus"]


                # FFmpeg command with audio re-encoding
                #TODO: expose audio quality options if format widgets makes it in
                #Reconsider forcing apad/shortest
                channels = audio['waveform'].size(1)
                min_audio_dur = total_frames_output / frame_rate + 1
                mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,
                            "-ar", str(audio['sample_rate']), "-ac", str(channels),
                            "-f", "f32le", "-i", "-", "-c:v", "copy"] \
                            + video_format["audio_pass"] \
                            + ["-af", "apad=whole_dur="+str(min_audio_dur),
                               "-shortest", output_file_with_audio_path]

                audio_data = audio['waveform'].squeeze(0).transpose(0,1) \
                        .numpy().tobytes()
                try:
                    res = subprocess.run(mux_args, input=audio_data,
                                         env=env, capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode("utf-8"))
                if res.stderr:
                    print(res.stderr.decode("utf-8"), end="", file=sys.stderr)
                output_files.append(output_file_with_audio_path)
                #Return this file with audio to the webui.
                #It will be muted unless opened or saved with right click
                file = output_file_with_audio

        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_output else "temp",
                "format": format,
                "frame_rate": frame_rate,
            }
        ]
        if num_frames == 1 and 'png' in format and '%03d' in file:
            previews[0]['format'] = 'image/png'
            previews[0]['filename'] = file.replace('%03d', '001')
        return {"result": ((save_output, output_files),)}

class LP_Engine:
    pipeline = None
    detect_model = None
    mask_img = None
    temp_img_idx = 0

    def get_temp_img_name(self):
        self.temp_img_idx += 1
        return "expression_edit_preview" + str(self.temp_img_idx) + ".png"

    def download_model(_, file_path, model_url):
        print('AdvancedLivePortrait: Downloading model...')
        response = requests.get(model_url, stream=True)
        try:
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte

                # tqdm will display a progress bar
                with open(file_path, 'wb') as file, tqdm(
                        desc='Downloading',
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(block_size):
                        bar.update(len(data))
                        file.write(data)

        except requests.exceptions.RequestException as err:
            print('AdvancedLivePortrait: Model download failed: {err}')
            print(f'AdvancedLivePortrait: Download it manually from: {model_url}')
            print(f'AdvancedLivePortrait: And put it in {file_path}')
        except Exception as e:
            print(f'AdvancedLivePortrait: An unexpected error occurred: {e}')

    def remove_ddp_dumplicate_key(_, state_dict):
        state_dict_new = OrderedDict()
        for key in state_dict.keys():
            state_dict_new[key.replace('module.', '')] = state_dict[key]
        return state_dict_new

    def filter_for_model(_, checkpoint, prefix):
        filtered_checkpoint = {key.replace(prefix + "_module.", ""): value for key, value in checkpoint.items() if
                               key.startswith(prefix)}
        return filtered_checkpoint

    def load_model(self, model_config, model_type):

        device = get_device()

        if model_type == 'stitching_retargeting_module':
            ckpt_path = os.path.join(get_model_dir("liveportrait"), "retargeting_models", model_type + ".pth")
        else:
            ckpt_path = os.path.join(get_model_dir("liveportrait"), "base_models", model_type + ".pth")

        is_safetensors = None
        if os.path.isfile(ckpt_path) == False:
            is_safetensors = True
            ckpt_path = os.path.join(get_model_dir("liveportrait"), model_type + ".safetensors")
            if os.path.isfile(ckpt_path) == False:
                self.download_model(ckpt_path,
                "https://huggingface.co/Kijai/LivePortrait_safetensors/resolve/main/" + model_type + ".safetensors")
        model_params = model_config['model_params'][f'{model_type}_params']
        if model_type == 'appearance_feature_extractor':
            model = AppearanceFeatureExtractor(**model_params).to(device)
        elif model_type == 'motion_extractor':
            model = MotionExtractor(**model_params).to(device)
        elif model_type == 'warping_module':
            model = WarpingNetwork(**model_params).to(device)
        elif model_type == 'spade_generator':
            model = SPADEDecoder(**model_params).to(device)
        elif model_type == 'stitching_retargeting_module':
            # Special handling for stitching and retargeting module
            config = model_config['model_params']['stitching_retargeting_module_params']
            checkpoint = load_torch_file(ckpt_path)

            stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
            if is_safetensors:
                stitcher.load_state_dict(self.filter_for_model(checkpoint, 'retarget_shoulder'))
            else:
                stitcher.load_state_dict(self.remove_ddp_dumplicate_key(checkpoint['retarget_shoulder']))
            stitcher = stitcher.to(device)
            stitcher.eval()

            return {
                'stitching': stitcher,
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")


        model.load_state_dict(load_torch_file(ckpt_path))
        model.eval()
        return model

    def load_models(self):
        model_path = get_model_dir("liveportrait")
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        model_config_path = os.path.join(current_directory, 'LivePortrait', 'config', 'models.yaml')
        model_config = yaml.safe_load(open(model_config_path, 'r'))

        appearance_feature_extractor = self.load_model(model_config, 'appearance_feature_extractor')
        motion_extractor = self.load_model(model_config, 'motion_extractor')
        warping_module = self.load_model(model_config, 'warping_module')
        spade_generator = self.load_model(model_config, 'spade_generator')
        stitching_retargeting_module = self.load_model(model_config, 'stitching_retargeting_module')

        self.pipeline = LivePortraitWrapper(InferenceConfig(), appearance_feature_extractor, motion_extractor, warping_module, spade_generator, stitching_retargeting_module)

    def get_detect_model(self):
        if self.detect_model == None:
            model_dir = get_model_dir("ultralytics")
            if not os.path.exists(model_dir): os.mkdir(model_dir)
            model_path = os.path.join(model_dir, "face_yolov8n.pt")
            if not os.path.exists(model_path):
                self.download_model(model_path, "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt")
            self.detect_model = YOLO(model_path)

        return self.detect_model

    def resize_image_to_stride(self,image_tensor, stride=32):
        # Get image height and width
        _, _, h, w = image_tensor.shape

        # Resize to a size divisible by the stride
        new_h = (h // stride) * stride
        new_w = (w // stride) * stride

        # Convert tensor to NumPy and resize
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (B, C, H, W) -> (H, W, C)
        resized_image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert the resized image back into a tensor
        resized_image_tensor = torch.from_numpy(resized_image_np).permute(2, 0, 1).unsqueeze(0).float()  # (H, W, C) -> (B, C, H, W)

        return resized_image_tensor

    def get_face_bboxes(self, image_rgb):
        detect_model = self.get_detect_model()

        # Check if images are loaded correctly
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("The input image is empty or invalid.")

        # Ensure that the image is 3 channel (RGB)
        if len(image_rgb.shape) == 2:  # If the image is grayscale, convert it to 3 channels
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)

        elif len(image_rgb.shape) == 3 and image_rgb.shape[2] == 1:  # Convert 1 channel (monochrome image) to 3 channels
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)

        # Check that the images are of the correct dimension and expand them to 4 dimensions (as expected by YOLO in batches)
        if len(image_rgb.shape) == 3:
            image_rgb = np.expand_dims(image_rgb, axis=0)  # (height, width, 3) -> (1, height, width, 3)

        # Dimension (batch_size, channels, height, width) Convert to the format
        image_rgb = torch.from_numpy(image_rgb).permute(0, 3, 1, 2).float()  # (1, height, width, 3) -> (1, 3, height, width)

        # Normalize pixel values ​​in the range 0-255 to 0-1
        image_rgb /= 255.0

        # Check size for debugging purposes
        print(f"Image dimensions before YOLO: {image_rgb.shape}")

        image_rgb = self.resize_image_to_stride(image_rgb, stride=32)


        pred = detect_model(image_rgb, conf=0.7, device="")  # Input to the YOLO model
        print("predicted all boxes: ", len(pred[0].boxes))
        return pred[0].boxes.xyxy.cpu().numpy()  # Returns the detected bounding box.

    def detect_face(self, image_tensor, crop_factor, sort = True):
        source_image_np = (image_tensor * 255).byte().cpu().numpy()
        img_rgb = source_image_np[0]
        bboxes = self.get_face_bboxes(img_rgb)
        w, h = get_rgb_size(img_rgb)

        print(f"w, h:{w, h}")
        result_boxes = []
        cx = w / 2
        min_diff = w
        best_box = None
        for x1, y1, x2, y2 in bboxes:
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            if bbox_w < 30: continue
            # diff = abs(cx - (x1 + bbox_w / 2))
            # if diff < min_diff:
            #     best_box = [x1, y1, x2, y2]
            #     print(f"diff, min_diff, best_box:{diff, min_diff, best_box}")
            #     min_diff = diff

        # if best_box == None:
        #     print("Failed to detect face!!")
        #     return [0, 0, w, h]

            # x1, y1, x2, y2 = best_box

        #for x1, y1, x2, y2 in bboxes:
            # bbox_w = x2 - x1
            # bbox_h = y2 - y1

            crop_w = bbox_w * crop_factor
            crop_h = bbox_h * crop_factor

            crop_w = max(crop_h, crop_w)
            crop_h = crop_w

            kernel_x = int(x1 + bbox_w / 2)
            kernel_y = int(y1 + bbox_h / 2)

            new_x1 = int(kernel_x - crop_w / 2)
            new_x2 = int(kernel_x + crop_w / 2)
            new_y1 = int(kernel_y - crop_h / 2)
            new_y2 = int(kernel_y + crop_h / 2)

        # if not sort:
        #     return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

            if sort:
                if new_x1 < 0:
                    new_x2 -= new_x1
                    new_x1 = 0
                elif w < new_x2:
                    new_x1 -= (new_x2 - w)
                    new_x2 = w
                    if new_x1 < 0:
                        new_x2 -= new_x1
                        new_x1 = 0

                if new_y1 < 0:
                    new_y2 -= new_y1
                    new_y1 = 0
                elif h < new_y2:
                    new_y1 -= (new_y2 - h)
                    new_y2 = h
                    if new_y1 < 0:
                        new_y2 -= new_y1
                        new_y1 = 0

                if w < new_x2 and h < new_y2:
                    over_x = new_x2 - w
                    over_y = new_y2 - h
                    over_min = min(over_x, over_y)
                    new_x2 -= over_min
                    new_y2 -= over_min
            result_boxes.append([int(new_x1), int(new_y1), int(new_x2), int(new_y2)])
        print(result_boxes)
        return result_boxes#[int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


    def calc_face_region(self, square, dsize):
        region = copy.deepcopy(square)
        is_changed = False
        if dsize[0] < region[2]:
            region[2] = dsize[0]
            is_changed = True
        if dsize[1] < region[3]:
            region[3] = dsize[1]
            is_changed = True

        return region, is_changed

    def expand_img(self, rgb_img, square):
        crop_trans_m = create_transform_matrix(max(-square[0], 0), max(-square[1], 0), 1, 1)
        new_img = cv2.warpAffine(rgb_img, crop_trans_m, (square[2] - square[0], square[3] - square[1]),
                                        cv2.INTER_LINEAR)
        return new_img

    def get_pipeline(self):
        if self.pipeline == None:
            print("Load pipeline...")
            self.load_models()

        return self.pipeline

    def prepare_src_image(self, img):
        h, w = img.shape[:2]
        input_shape = [256,256]
        if h != input_shape[0] or w != input_shape[1]:
            if 256 < h: interpolation = cv2.INTER_AREA
            else: interpolation = cv2.INTER_LINEAR
            x = cv2.resize(img, (input_shape[0], input_shape[1]), interpolation = interpolation)
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        x = np.clip(x, 0, 1)  # clip to 0~1
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.to(get_device())
        return x

    def GetMaskImg(self):
        if self.mask_img is None:
            path = os.path.join(current_directory, "LivePortrait/utils/resources/mask_template.png")
            self.mask_img = cv2.imread(path, cv2.IMREAD_COLOR)
            if self.mask_img is None:
                raise FileNotFoundError(f"Mask image not found at path: {path}")
        return self.mask_img


    def crop_face(self, img_rgb, crop_factor):
        crop_region = self.detect_face(img_rgb, crop_factor)
        face_region, is_changed = self.calc_face_region(crop_region, get_rgb_size(img_rgb))
        face_img = rgb_crop(img_rgb, face_region)
        if is_changed: face_img = self.expand_img(face_img, crop_region)
        return face_img

    def prepare_source(self, source_image, crop_region, is_video = False, tracking = False):
        print("Prepare source...")
        engine = self.get_pipeline()
        source_image_np = (source_image * 255).byte().cpu().numpy()
        img_rgb = source_image_np[0]

        psi_list = []
        for img_rgb in source_image_np:
            if tracking or len(psi_list) == 0:
                # crop_region = self.detect_face(img_rgb, crop_factor)
                face_region, is_changed = self.calc_face_region(crop_region, get_rgb_size(img_rgb))

                s_x = (face_region[2] - face_region[0]) / 512.
                s_y = (face_region[3] - face_region[1]) / 512.
                crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], s_x, s_y)
                mask_ori = cv2.warpAffine(self.GetMaskImg(), crop_trans_m, get_rgb_size(img_rgb), cv2.INTER_LINEAR)
                mask_ori = mask_ori.astype(np.float32) / 255.

                if is_changed:
                    s = (crop_region[2] - crop_region[0]) / 512.
                    crop_trans_m = create_transform_matrix(crop_region[0], crop_region[1], s, s)

            face_img = rgb_crop(img_rgb, face_region)
            if is_changed: face_img = self.expand_img(face_img, crop_region)
            i_s = self.prepare_src_image(face_img)
            x_s_info = engine.get_kp_info(i_s)
            f_s_user = engine.extract_feature_3d(i_s)
            x_s_user = engine.transform_keypoint(x_s_info)
            psi = PreparedSrcImg(img_rgb, crop_trans_m, x_s_info, f_s_user, x_s_user, mask_ori)
            if is_video == False:
                return psi
            psi_list.append(psi)

        return psi_list

    def prepare_driving_video(self, face_images):
        print("Prepare driving video...")
        pipeline = self.get_pipeline()
        f_img_np = (face_images * 255).byte().numpy()

        out_list = []
        for f_img in f_img_np:
            i_d = self.prepare_src_image(f_img)
            d_info = pipeline.get_kp_info(i_d)
            out_list.append(d_info)

        return out_list

    def calc_fe(_, x_d_new, eyes, eyebrow, wink, pupil_x, pupil_y, mouth, eee, woo, smile,
                rotate_pitch, rotate_yaw, rotate_roll):

        x_d_new[0, 20, 1] += smile * -0.01
        x_d_new[0, 14, 1] += smile * -0.02
        x_d_new[0, 17, 1] += smile * 0.0065
        x_d_new[0, 17, 2] += smile * 0.003
        x_d_new[0, 13, 1] += smile * -0.00275
        x_d_new[0, 16, 1] += smile * -0.00275
        x_d_new[0, 3, 1] += smile * -0.0035
        x_d_new[0, 7, 1] += smile * -0.0035

        x_d_new[0, 19, 1] += mouth * 0.001
        x_d_new[0, 19, 2] += mouth * 0.0001
        x_d_new[0, 17, 1] += mouth * -0.0001
        rotate_pitch -= mouth * 0.05

        x_d_new[0, 20, 2] += eee * -0.001
        x_d_new[0, 20, 1] += eee * -0.001
        #x_d_new[0, 19, 1] += eee * 0.0006
        x_d_new[0, 14, 1] += eee * -0.001

        x_d_new[0, 14, 1] += woo * 0.001
        x_d_new[0, 3, 1] += woo * -0.0005
        x_d_new[0, 7, 1] += woo * -0.0005
        x_d_new[0, 17, 2] += woo * -0.0005

        x_d_new[0, 11, 1] += wink * 0.001
        x_d_new[0, 13, 1] += wink * -0.0003
        x_d_new[0, 17, 0] += wink * 0.0003
        x_d_new[0, 17, 1] += wink * 0.0003
        x_d_new[0, 3, 1] += wink * -0.0003
        rotate_roll -= wink * 0.1
        rotate_yaw -= wink * 0.1

        if 0 < pupil_x:
            x_d_new[0, 11, 0] += pupil_x * 0.0007
            x_d_new[0, 15, 0] += pupil_x * 0.001
        else:
            x_d_new[0, 11, 0] += pupil_x * 0.001
            x_d_new[0, 15, 0] += pupil_x * 0.0007

        x_d_new[0, 11, 1] += pupil_y * -0.001
        x_d_new[0, 15, 1] += pupil_y * -0.001
        eyes -= pupil_y / 2.

        x_d_new[0, 11, 1] += eyes * -0.001
        x_d_new[0, 13, 1] += eyes * 0.0003
        x_d_new[0, 15, 1] += eyes * -0.001
        x_d_new[0, 16, 1] += eyes * 0.0003
        x_d_new[0, 1, 1] += eyes * -0.00025
        x_d_new[0, 2, 1] += eyes * 0.00025


        if 0 < eyebrow:
            x_d_new[0, 1, 1] += eyebrow * 0.001
            x_d_new[0, 2, 1] += eyebrow * -0.001
        else:
            x_d_new[0, 1, 0] += eyebrow * -0.001
            x_d_new[0, 2, 0] += eyebrow * 0.001
            x_d_new[0, 1, 1] += eyebrow * 0.0003
            x_d_new[0, 2, 1] += eyebrow * -0.0003


        return torch.Tensor([rotate_pitch, rotate_yaw, rotate_roll])
g_engine = LP_Engine()
class ExpressionSet:
    def __init__(self, erst = None, es = None):
        if es != None:
            self.e = copy.deepcopy(es.e)  # [:, :, :]
            self.r = copy.deepcopy(es.r)  # [:]
            self.s = copy.deepcopy(es.s)
            self.t = copy.deepcopy(es.t)
        elif erst != None:
            self.e = erst[0]
            self.r = erst[1]
            self.s = erst[2]
            self.t = erst[3]
        else:
            self.e = torch.from_numpy(np.zeros((1, 21, 3))).float().to(get_device())
            self.r = torch.Tensor([0, 0, 0])
            self.s = 0
            self.t = 0
    def div(self, value):
        self.e /= value
        self.r /= value
        self.s /= value
        self.t /= value
    def add(self, other):
        self.e += other.e
        self.r += other.r
        self.s += other.s
        self.t += other.t
    def sub(self, other):
        self.e -= other.e
        self.r -= other.r
        self.s -= other.s
        self.t -= other.t
    def mul(self, value):
        self.e *= value
        self.r *= value
        self.s *= value
        self.t *= value

    #def apply_ratio(self, ratio):        self.exp *= ratio
exp_data_dir = "./exp_data"
if os.path.isdir(exp_data_dir) == False:
    os.mkdir(exp_data_dir)
class SaveExpData:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "file_name": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {"save_exp": ("EXP_DATA",), }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_name",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"
    OUTPUT_NODE = True

    def run(self, file_name, save_exp:ExpressionSet=None):
        if save_exp == None or file_name == "":
            return file_name

        with open(os.path.join(exp_data_dir, file_name + ".exp"), "wb") as f:
            dill.dump(save_exp, f)
        return file_name

class LoadExpData:
    @classmethod
    def INPUT_TYPES(s):
        file_list = [os.path.splitext(file)[0] for file in os.listdir(exp_data_dir) if file.endswith('.exp')]
        return {"required": {
            "file_name": (sorted(file_list, key=str.lower),),
            "ratio": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
        },
        }

    RETURN_TYPES = ("EXP_DATA",)
    RETURN_NAMES = ("exp",)
    FUNCTION = "run"
    CATEGORY = "AdvancedLivePortrait"

    def run(self, file_name, ratio):
        # es = ExpressionSet()
        with open(os.path.join(exp_data_dir, file_name + ".exp"), 'rb') as f:
            es = dill.load(f)
        es.mul(ratio)
        return (es,)
class ExpressionSet:
    def __init__(self, erst = None, es = None):
        if es != None:
            self.e = copy.deepcopy(es.e)  # [:, :, :]
            self.r = copy.deepcopy(es.r)  # [:]
            self.s = copy.deepcopy(es.s)
            self.t = copy.deepcopy(es.t)
        elif erst != None:
            self.e = erst[0]
            self.r = erst[1]
            self.s = erst[2]
            self.t = erst[3]
        else:
            self.e = torch.from_numpy(np.zeros((1, 21, 3))).float().to(get_device())
            self.r = torch.Tensor([0, 0, 0])
            self.s = 0
            self.t = 0
    def div(self, value):
        self.e /= value
        self.r /= value
        self.s /= value
        self.t /= value
    def add(self, other):
        self.e += other.e
        self.r += other.r
        self.s += other.s
        self.t += other.t
    def sub(self, other):
        self.e -= other.e
        self.r -= other.r
        self.s -= other.s
        self.t -= other.t
    def mul(self, value):
        self.e *= value
        self.r *= value
        self.s *= value
        self.t *= value

    #def apply_ratio(self, ratio):        self.exp *= ratio

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time - start_time))
        return result

    return wrapper_fn


class Command:
    def __init__(self, es, change, keep):
        self.es:ExpressionSet = es
        self.change = change
        self.keep = keep

crop_factor_default = 1.7
crop_factor_min = 1.5
crop_factor_max = 2.5

class AdvancedLivePortrait:
    def __init__(self, engine):
        self.src_images = None
        self.driving_images = None
        self.crop_factor = None
        self.g_engine = g_engine
        self.pipeline = g_engine.get_pipeline()  # Load the pipeline once

    def parsing_command(self, command, motoin_link):
        command.replace(' ', '')
        # if command == '': return
        lines = command.split('\n')

        cmd_list = []

        total_length = 0

        i = 0
        #old_es = None
        for line in lines:
            i += 1
            if line == '': continue
            try:
                cmds = line.split('=')
                idx = int(cmds[0])
                if idx == 0: es = ExpressionSet()
                else: es = ExpressionSet(es = motoin_link[idx])
                cmds = cmds[1].split(':')
                change = int(cmds[0])
                keep = int(cmds[1])
            except:
                assert False, f"(AdvancedLivePortrait) Command Err Line {i}: {line}"


                return None, None

            total_length += change + keep
            es.div(change)
            cmd_list.append(Command(es, change, keep))

        return cmd_list, total_length


    def run(self, retargeting_eyes, retargeting_mouth, turn_on, tracking_src_vid, animate_without_vid, command, crop_factor,
            src_images=None, driving_images=None, motion_link=None):
        if turn_on == False: return (None,None)
        src_length = 1

        if src_images == None:
            if motion_link != None:
                self.psi_list = [motion_link[0]]
            else: return (None,None)

        if src_images != None:
            src_length = len(src_images)
            if id(src_images) != id(self.src_images) or self.crop_factor != crop_factor:
                self.crop_factor = crop_factor
                self.src_images = src_images
                if 1 < src_length:
                    self.psi_list = self.g_engine.prepare_source(src_images, crop_factor, True, tracking_src_vid)
                else:
                    self.psi_list = [self.g_engine.prepare_source(src_images, crop_factor)]


        cmd_list, cmd_length = self.parsing_command(command, motion_link)
        if cmd_list == None: return (None,None)
        cmd_idx = 0

        driving_length = 0
        if driving_images is not None:
            if id(driving_images) != id(self.driving_images):
                self.driving_images = driving_images
                self.driving_values = self.g_engine.prepare_driving_video(driving_images)
            driving_length = len(self.driving_values)

        total_length = max(driving_length, src_length)

        if animate_without_vid:
            total_length = max(total_length, cmd_length)

        c_i_es = ExpressionSet()
        c_o_es = ExpressionSet()
        d_0_es = None
        out_list = []

        psi = None
        # pipeline = g_engine.get_pipeline()
        for i in range(total_length):

            if i < src_length:
                psi = self.psi_list[i]
                s_info = psi.x_s_info
                s_es = ExpressionSet(erst=(s_info['kp'] + s_info['exp'], torch.Tensor([0, 0, 0]), s_info['scale'], s_info['t']))

            new_es = ExpressionSet(es = s_es)

            if i < cmd_length:
                cmd = cmd_list[cmd_idx]
                if 0 < cmd.change:
                    cmd.change -= 1
                    c_i_es.add(cmd.es)
                    c_i_es.sub(c_o_es)
                elif 0 < cmd.keep:
                    cmd.keep -= 1

                new_es.add(c_i_es)

                if cmd.change == 0 and cmd.keep == 0:
                    cmd_idx += 1
                    if cmd_idx < len(cmd_list):
                        c_o_es = ExpressionSet(es = c_i_es)
                        cmd = cmd_list[cmd_idx]
                        c_o_es.div(cmd.change)
            elif 0 < cmd_length:
                new_es.add(c_i_es)

            if i < driving_length:
                d_i_info = self.driving_values[i]
                d_i_r = torch.Tensor([d_i_info['pitch'], d_i_info['yaw'], d_i_info['roll']])#.float().to(device="cuda:0")

                if d_0_es is None:
                    d_0_es = ExpressionSet(erst = (d_i_info['exp'], d_i_r, d_i_info['scale'], d_i_info['t']))

                    retargeting(s_es.e, d_0_es.e, retargeting_eyes, (11, 13, 15, 16))
                    retargeting(s_es.e, d_0_es.e, retargeting_mouth, (14, 17, 19, 20))

                new_es.e += d_i_info['exp'] - d_0_es.e
                new_es.r += d_i_r - d_0_es.r
                new_es.t += d_i_info['t'] - d_0_es.t

            r_new = get_rotation_matrix(
                s_info['pitch'] + new_es.r[0], s_info['yaw'] + new_es.r[1], s_info['roll'] + new_es.r[2])
            d_new = new_es.s * (new_es.e @ r_new) + new_es.t
            d_new = self.pipeline.stitching(psi.x_s_user, d_new)
            crop_out = self.pipeline.warp_decode(psi.f_s_user, psi.x_s_user, d_new)
            crop_out = self.pipeline.parse_output(crop_out['out'])[0]

            crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb),
                                                cv2.INTER_LINEAR)
            out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(
                np.uint8)
            out_list.append(out)

        if len(out_list) == 0: return (None,)

        out_imgs = torch.cat([pil2tensor(img_rgb) for img_rgb in out_list])
        return (out_imgs,)

class ExpressionEditor:
    def __init__(self, g_engine):
        self.sample_image = None
        self.src_image = None
        self.crop_factor = None
        self.g_engine = g_engine
        self.pipeline = g_engine.get_pipeline()  # Load the pipeline once
        self.psi = None  # To store the processed source image
        # self.crop_region = crop_region
        print("ExpressionEditor model loaded")

    def run(self, rotate_pitch, rotate_yaw, rotate_roll, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile,
            src_ratio, sample_ratio, sample_parts, crop_factor, crop_region, src_image=None, sample_image=None, motion_link=None, add_exp=None):
        
        rotate_yaw = -rotate_yaw

        new_editor_link = None
        if motion_link is not None:
            self.psi = motion_link[0]
            new_editor_link = motion_link.copy()
        elif src_image is not None:
            if id(src_image) != id(self.src_image) or self.crop_factor != crop_factor:
                self.crop_factor = crop_factor
                self.psi = self.g_engine.prepare_source(src_image, crop_region)
                self.src_image = src_image
            new_editor_link = []
            new_editor_link.append(self.psi)
        else:
            return (None, None)

        psi = self.psi
        s_info = psi.x_s_info
        s_exp = s_info['exp'] * src_ratio
        s_exp[0, 5] = s_info['exp'][0, 5]
        s_exp += s_info['kp']

        es = ExpressionSet()
        es_new = ExpressionSet()
        es_new.e = s_info['exp']
        if sample_image is not None:
            if id(self.sample_image) != id(sample_image):
                self.sample_image = sample_image
                d_image_np = (sample_image * 255).cpu().byte().numpy()
                d_face = self.g_engine.crop_face(d_image_np[0], 1.7)
                i_d = self.g_engine.prepare_src_image(d_face)
                self.d_info = self.pipeline.get_kp_info(i_d)
                self.d_info['exp'][0, 5, 0] = 0
                self.d_info['exp'][0, 5, 1] = 0

            # "OnlyExpression", "OnlyRotation", "OnlyMouth", "OnlyEyes", "All"
            if sample_parts == "OnlyExpression" or sample_parts == "All":
                es.e += self.d_info['exp'] * sample_ratio
            if sample_parts == "OnlyRotation" or sample_parts == "All":
                rotate_pitch += self.d_info['pitch'] * sample_ratio
                rotate_yaw += self.d_info['yaw'] * sample_ratio
                rotate_roll += self.d_info['roll'] * sample_ratio
            elif sample_parts == "OnlyMouth":
                retargeting(es.e, self.d_info['exp'], sample_ratio, (14, 17, 19, 20))
            elif sample_parts == "OnlyEyes":
                retargeting(es.e, self.d_info['exp'], sample_ratio, (1, 2, 11, 13, 15, 16))

        es.r = self.g_engine.calc_fe(es.e, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile,
                                  rotate_pitch, rotate_yaw, rotate_roll)

        if add_exp is not None:
            es.add(add_exp)

        new_rotate = get_rotation_matrix(s_info['pitch'] + es.r[0], s_info['yaw'] + es.r[1],
                                         s_info['roll'] + es.r[2])
        x_d_new = (s_info['scale'] * (1 + es.s)) * ((s_exp + es.e) @ new_rotate) + s_info['t']

        x_d_new = self.pipeline.stitching(psi.x_s_user, x_d_new)

        crop_out = self.pipeline.warp_decode(psi.f_s_user, psi.x_s_user, x_d_new)
        crop_out = self.pipeline.parse_output(crop_out['out'])[0]

        crop_with_fullsize = cv2.warpAffine(crop_out, psi.crop_trans_m, get_rgb_size(psi.src_rgb), cv2.INTER_LINEAR)
        out = np.clip(psi.mask_ori * crop_with_fullsize + (1 - psi.mask_ori) * psi.src_rgb, 0, 255).astype(np.uint8)

        out_img = pil2tensor(out)

        filename = self.g_engine.get_temp_img_name()  # Filename for saving the image
        results = [{"filename": filename, "type": "temp"}]

        new_editor_link.append(es)

        return {"result": (out_img, new_editor_link, es_new)}
def display_detected_faces(image, face_regions):
    """
    Display image with numbered boxes around detected faces
    
    Args:
        image: PIL Image
        face_regions: List of face coordinates [x1, y1, x2, y2]
    Returns:
        PIL Image with drawn face boxes
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # Create a copy of the image to draw on
    display_img = image.copy()
    draw = ImageDraw.Draw(display_img)
    
    # Define colors for the boxes and text
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'orange']
    
    # Try to load a font, fall back to default if not available
    try:
        # For macOS
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    except:
        try:
            # For Windows
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fallback to default
            font = ImageFont.load_default()
    
    # Draw boxes and numbers for each detected face
    for i, region in enumerate(face_regions):
        x1, y1, x2, y2 = region
        color = colors[i % len(colors)]  # Cycle through colors for different faces
        
        # Draw rectangle around face
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Add face number
        label = f"Face {i}"
        text_bbox = draw.textbbox((x1, y1-25), label, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], 
                      fill=color)
        draw.text((x1, y1-25), label, fill='white', font=font)
    
    return display_img

engine = LP_Engine()
exp_editor = ExpressionEditor(engine)
# crop_region = engine.detect_face(img_rgb, crop_factor)
def AdvancedLivePortrait_execution(exp_editor, src_image, parameters, crop_region, sample_image=None, add_exp=None):
    # Specifies the path to the input image.

    # if not os.path.exists(os.path.dirname(output_image_path)):
    #     os.makedirs(os.path.dirname(output_image_path))

    # Load an image containing a face
    # image = Image.open(input_image_path)
    src_image = src_image.convert("RGB")
    src_img_tensor = pil2tensor(src_image).to(get_device())  # Convert the image to a tensor and transfer it to the GPU
    if sample_image is not None:
        sample_image = sample_image.convert("RGB")
        sample_image = pil2tensor(sample_image).to(get_device())
    # print(img_tensor.shape)
    if add_exp is not None:
        add_exp = add_exp
    # print("load image")
    # Create an instance of the LP_Engine class
    # engine = LP_Engine()

    

    # Settings for changing facial expressions
    rotate_pitch = parameters[0]  # Vertical face rotation (front to back)
    rotate_yaw = parameters[1]    # Lateral face rotation (left/right)
    rotate_roll = parameters[2]   # Face tilt
    blink = parameters[3]         # Degree of blinking
    eyebrow = parameters[4]       # Eyebrow movement
    wink = parameters[5]          # One-Eyed Wink
    pupil_x = parameters[6]       # Left and right movement of pupils
    pupil_y = parameters[7]       # Up and down movement of the pupils
    aaa = parameters[8]           # Mouth opening movement
    eee = parameters[9]           # The movement of opening your mouth to make the sound "i"
    woo = parameters[10]          # The movement of opening your mouth to make the sound "u"
    smile = parameters[11]        # Smile level (0.5 = 50% smile)

    print("start running")

    # Edit facial expressions and get the resulting image
    result = exp_editor.run(
        rotate_pitch, rotate_yaw, rotate_roll,
        blink, eyebrow, wink, pupil_x, pupil_y,
        aaa, eee, woo, smile,
        src_ratio=1, sample_ratio=1,
        sample_parts="All",  # Editing the overall expression
        crop_factor=1.7,
        crop_region = crop_region,
        src_image=src_img_tensor,
        sample_image = sample_image,
        motion_link = None,
        add_exp = add_exp
    )

    # Expand the results
    edited_img, motion_link, expression_data = result["result"]

    print("finish running")

    # Save the resulting image
    edited_image_pil = tensor2pil(edited_img)  # Convert tensor to PIL image
    return edited_image_pil, motion_link, expression_data
input_image_path = "multiple faces.jpg"  
output_image_path = "./outputs/edited_image.png"
sample_image =  "./sample/fang.jpg"
# Settings for changing facial expressions
rotate_pitch = 0     # Vertical face rotation (front and back) "rotate_pitch": ("FLOAT", {"default": 0, "min": -20, "max": 20, "step": 0.5, "display": display})
rotate_yaw = 0       # Horizontal face rotation (left and right) "rotate_yaw": ("FLOAT", {"default": 0, "min": -20, "max": 20, "step": 0.5, "display": display})
rotate_roll = 0      # Face tilt "rotate_roll": ("FLOAT", {"default": 0, "min": -20, "max": 20, "step": 0.5, "display": display}),
blink = 0            #Blink intensity "blink": ("FLOAT", {"default": 0, "min": -20, "max": 5, "step": 0.5, "display": display}),
eyebrow = 0          #Eyebrow movement "eyebrow": ("FLOAT", {"default": 0, "min": -10, "max": 15, "step": 0.5, "display": display}),
wink = 0             #One-eyed wink  "wink": ("FLOAT", {"default": 0, "min": 0, "max": 25, "step": 0.5, "display": display}),
pupil_x = 0          # Pupil left and right movement "pupil_x": ("FLOAT", {"default": 0, "min": -15, "max": 15, "step": 0.5, "display": display}),
pupil_y = 0          # Pupil up and down movement  "pupil_y": ("FLOAT", {"default": 0, "min": -15, "max": 15, "step": 0.5, "display": display}),
aaa = 0              # Mouth opening action  "aaa": ("FLOAT", {"default": 0, "min": -30, "max": 120, "step": 1, "display": display}),
eee = 0              # Mouth making "e" sound  "eee": ("FLOAT", {"default": 0, "min": -20, "max": 15, "step": 0.2, "display": display}),
woo = 0              # Mouth making "u" sound  "woo": ("FLOAT", {"default": 0, "min": -20, "max": 15, "step": 0.2, "display": display}),
smile = 2            # Smile intensity (0.5 = 50% smile) "smile": ("FLOAT", {"default": 0, "min": -0.3, "max": 1.3, "step": 0.01, "display": display}),
                


input_img = Image.open(input_image_path)
sample_image = Image.open(sample_image)
src_image = input_img.convert("RGB")
src_img_tensor = pil2tensor(src_image).to(get_device())
sample_image = sample_image.convert("RGB")
sample_image = pil2tensor(sample_image).to(get_device())
# source_image_np = (src_img_tensor * 255).byte().cpu().numpy()
# img_rgb = source_image_np[0]
face_list = engine.detect_face(src_img_tensor, crop_factor =1.7)
parameters = [rotate_pitch, rotate_yaw, rotate_roll, blink, eyebrow, wink, pupil_x, pupil_y, aaa, eee, woo, smile]
edited_img, motion_link, expression_data = AdvancedLivePortrait_execution(exp_editor, input_img, parameters, face_list[2], sample_image=None, add_exp=None)
marked_image = display_detected_faces(edited_img, face_list)
marked_image.show()
# edited_img.show()

# def Create_gif(adv_editor, vid_editor, motion_link):
#     retargeting_eyes = 0.00
#     retargeting_mouth = 0.00
#     crop_factor = crop_factor_default
#     turn_on = True
#     tracking_src_vid = False
#     animate_without_vid = True
#     command = "1=5:5\n0=3:5"
#     out_images = adv_editor.run(retargeting_eyes, retargeting_mouth, turn_on, tracking_src_vid, animate_without_vid, command, crop_factor,
#             src_images=None, driving_images=None, motion_link=motion_link)
    
#     out_images = out_images[0]
#     # print(out_images)
#     # vid_com = VideoCombine()
#     # Define other parameters
#     frame_rate = 24.0  # Example frame rate
#     loop_count = 0      # 0 for infinite loop in GIFs
#     filename_prefix = "test"
#     format = "image/gif"  # Can be "image/gif" or other supported formats
#     pingpong = False
#     save_output = True
#     prompt = None#{"example_prompt_key": "example_prompt_value"}  # Replace with actual prompt
#     extra_pnginfo = None#{"key1": "value1", "key2": "value2"}    # Replace as needed
#     audio = None  # Replace with actual audio data if available
#     unique_id = "unique_identifier"  # Replace with actual unique ID if using meta_batch
#     manual_format_widgets = None  # Replace or define as needed
#     meta_batch = None  # Replace with actual VHS_BatchManager instance if needed
#     vae = None  # Replace with actual VAE instance if needed
#     video_res = vid_editor.combine_video(
#         frame_rate= frame_rate,
#         loop_count = loop_count,
#         images=out_images,
#         latents = None,
#         filename_prefix= filename_prefix,
#         format=format,
#         pingpong = pingpong,
#         save_output = save_output,
#         prompt=prompt,
#         extra_pnginfo=extra_pnginfo,
#         audio=audio,
#         unique_id=unique_id,
#         manual_format_widgets=manual_format_widgets,
#         meta_batch=meta_batch,
#         vae=vae

#     )

#     return video_res['result'][0][1][1]