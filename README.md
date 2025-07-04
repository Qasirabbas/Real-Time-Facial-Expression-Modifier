# Advanced Live Portrait

An advanced facial expression manipulation and animation system using deep learning. This project provides both manual expression editing and automatic expression imitation capabilities with a modern web interface.

## 🌟 Features

### Manual Expression Editor
- **Precise Control**: Fine-tune facial expressions with intuitive sliders
- **Multi-face Support**: Detect and edit multiple faces in a single image
- **Real-time Preview**: See changes instantly as you adjust parameters
- **Expression Categories**: 
  - Head movements (pitch, yaw, roll)
  - Eye controls (blink, eyebrow, wink, pupil movement)
  - Mouth expressions (open mouth, vowel sounds, smile)

### Automatic Expression Imitation
- **One-click Imitation**: Transfer expressions from reference images
- **Face Selection**: Choose which face to edit in multi-face images
- **High Quality Results**: Maintain image quality while changing expressions

### Animation Generation
- **GIF Creation**: Generate animated sequences from edited expressions
- **Smooth Transitions**: Create natural-looking facial animations
- **Export Options**: Download results as images or animated GIFs

### Modern Web Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Drag & Drop**: Easy image uploading with drag and drop support
- **Real-time Face Detection**: Automatic face detection with visual feedback
- **Dark Theme**: Modern, eye-friendly interface

## 🛠️ Technology Stack

- **Backend**: FastAPI with Python
- **Frontend**: Vanilla JavaScript with modern CSS
- **Deep Learning**: PyTorch with LivePortrait models
- **Face Detection**: YOLO-based face detection
- **Image Processing**: OpenCV, PIL
- **Authentication**: JWT-based user authentication

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space for models

### Python Dependencies
```
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
Pillow>=8.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
numpy>=1.21.0
pyyaml>=5.4.0
requests>=2.25.0
tqdm>=4.62.0
ultralytics>=8.0.0
safetensors>=0.3.0
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/advanced-live-portrait.git
cd advanced-live-portrait
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models
The application will automatically download required models on first run:
- LivePortrait models (~2GB)
- YOLO face detection model (~6MB)

Models are stored in `./models/` directory.

### 5. Setup Directories
```bash
mkdir -p models/liveportrait/base_models
mkdir -p models/liveportrait/retargeting_models
mkdir -p models/ultralytics
mkdir -p outputs
mkdir -p static
mkdir -p templates
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:
```env
# Server Configuration
HOST=0.0.0.0
PORT=4001
LOG_LEVEL=info

# Authentication
AUTH_URL=http://192.168.1.2:5009/check_auth
SECRET_KEY=your-secret-key-here

# Storage
BASE_FOLDER=/path/to/user/data
TEMP_FOLDER=/path/to/temp/files

# Model Configuration
DEVICE=cuda  # or cpu
MODEL_CACHE_SIZE=3
```

### Model Configuration
Edit `LivePortrait/config/models.yaml` to customize model parameters.

## 🏃 Running the Application

### Development Mode
```bash
python main.py
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 4001 --workers 1
```

### Using Docker
```bash
docker build -t advanced-live-portrait .
docker run -p 4001:4001 -v ./models:/app/models advanced-live-portrait
```

## 📖 Usage

### Manual Mode
1. Upload an image with faces
2. Select a face by clicking on the detection box
3. Use sliders to adjust facial expressions:
   - **Head**: Control head rotation and tilt
   - **Eyes**: Adjust eye movements and expressions
   - **Mouth**: Modify mouth shape and expressions
4. Download the edited image or create an animation

### Automatic Mode
1. Upload a source image (the face you want to edit)
2. Upload a reference image (the expression you want to copy)
3. Select which face to edit if multiple faces are detected
4. Click "一键模仿" (One-click Imitation)
5. Download the result or create an animation

### API Endpoints

#### Image Upload
```http
POST /expression/upload
Content-Type: multipart/form-data

file: [image file]
```

#### Face Detection
```http
GET /expression/detect-faces/{image_id}
```

#### Expression Editing
```http
POST /expression/edit
Content-Type: multipart/form-data

image_id: string
face_index: integer
rotate_pitch: float
rotate_yaw: float
rotate_roll: float
blink: float
eyebrow: float
wink: float
pupil_x: float
pupil_y: float
aaa: float
eee: float
woo: float
smile: float
```

#### Expression Imitation
```http
POST /expression/process-imitation
Content-Type: multipart/form-data

source_image: [image file]
sample_image: [image file]
face_index: integer
```

#### Animation Generation
```http
POST /expression/create-gif
```

## 🎨 Customization

### Adding New Expression Controls
1. Add slider HTML in `templates/index3.html`
2. Update the slider handling in JavaScript
3. Modify the `calc_fe` function in `Advance_Live_Portrait.py`
4. Update the API endpoint parameter handling

### Styling Customization
- Modify CSS variables in `:root` section
- Update component styles in the `<style>` section
- Add custom animations and transitions

### Model Customization
- Replace model files in `./models/` directory
- Update model configuration in `LivePortrait/config/models.yaml`
- Modify loading functions in `LP_Engine` class

## 🔍 Troubleshooting

### Common Issues

#### GPU Memory Issues
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU mode by setting `DEVICE=cpu`

#### Model Download Failures
```
Model download failed
```
**Solution**: Download models manually from provided URLs and place in correct directories

#### Face Detection Not Working
```
No faces detected
```
**Solution**: Ensure image has clear, front-facing faces and good lighting

#### Slow Performance
**Solutions**:
- Use GPU acceleration
- Reduce image resolution
- Close other applications
- Use smaller batch sizes

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance

### Benchmarks
- **Face Detection**: ~0.1-0.5s per image
- **Expression Editing**: ~1-3s per face
- **Animation Generation**: ~5-15s for 24 frames
- **Memory Usage**: ~2-6GB GPU memory

### Optimization Tips
- Use GPU acceleration when available
- Batch process multiple images
- Cache model weights
- Use appropriate image resolutions (512x512 recommended)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Test on different devices and browsers

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LivePortrait](https://github.com/KwaiVGI/LivePortrait) - Core facial animation technology
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Face detection
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [PyTorch](https://pytorch.org/) - Deep learning framework

## 📞 Support

- **Documentation**: [Wiki](https://github.com/yourusername/advanced-live-portrait/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/advanced-live-portrait/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/advanced-live-portrait/discussions)

## 🗺️ Roadmap

- [ ] Real-time video processing
- [ ] 3D face model support
- [ ] Voice-driven animation
- [ ] Mobile app development
- [ ] Cloud deployment options
- [ ] Advanced animation presets
- [ ] Batch processing interface
- [ ] Plugin system for custom effects

## 📈 Changelog

### Version 1.0.0 (Current)
- Initial release
- Manual expression editing
- Automatic expression imitation
- Web interface
- Multi-face support
- Animation generation

---

⭐ If you find this project helpful, please give it a star on GitHub!
