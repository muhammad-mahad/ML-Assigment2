# 🚀 QUICK START GUIDE - Face Detection Neural Network

## One-Command Setup & Training

### Option 1: Cross-Platform Python Script (Recommended)
Works on Windows, Mac, and Linux:

```bash
# Interactive menu
python automate.py
```

### Option 2: Platform-Specific Scripts

#### Windows:
```batch
run.bat
```

#### Mac/Linux:
```bash
chmod +x run.sh
./run.sh
```

---

## ⚡ What These Scripts Do

1. **Create Virtual Environment** - Isolates project dependencies
2. **Install Requirements** - NumPy, Pillow, Matplotlib, OpenCV (optional)
3. **Setup Directories** - Creates all necessary folders
4. **Generate Dataset** - Creates synthetic images if none exist
5. **Train Model** - Runs complete training pipeline
6. **Test System** - Verifies all components work
7. **Test Predictions** - Confirms model can make predictions

---

## 📦 Files Included

| File | Purpose | Usage |
|------|---------|-------|
| `requirements.txt` | Python dependencies | `pip install -r requirements.txt` |
| `run.bat` | Windows automation | Double-click or run in cmd |
| `run.sh` | Mac/Linux automation | `./run.sh` |
| `automate.py` | Cross-platform menu | `python automate.py` |

---

## 🎯 Usage Examples

### Interactive Setup (With Options)
```bash
# Shows menu with choices
python automate.py
```

### Manual Control
```bash
# Just train the model
python automate.py --train

# Just run tests
python automate.py --test
```

---

## 📊 Expected Output

After running the automated setup, you'll have:

✅ **Virtual environment** configured  
✅ **All dependencies** installed  
✅ **Dataset** prepared (synthetic if no real images)  
✅ **Model trained** with ~90%+ accuracy  
✅ **All tests** passing  
✅ **Ready for predictions**  

---

## 🖼️ Adding Your Own Images

For best results with real faces:

1. **Before running scripts**, add images to:
   - `face_images/` - Your face photos (30+ recommended)
   - `non_face_images/` - Non-face images (30+ recommended)

2. **Image requirements:**
   - Formats: JPG, JPEG, PNG
   - Any size (will be resized automatically)
   - Clear, well-lit photos work best

---

## 🔧 Troubleshooting

### "Python not found"
- Install Python 3.7+ from https://python.org
- Add Python to PATH during installation

### "Permission denied" (Mac/Linux)
```bash
chmod +x run.sh
chmod +x automate.py
```

### "Module not found"
```bash
# Activate virtual environment first
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# Then install requirements
pip install -r requirements.txt
```

### Low Accuracy
- Add more training images (50+ recommended)
- Ensure good image quality
- Check for mislabeled images
- Run training for more epochs

---

## 🎮 Using the Trained Model

After setup completes:

```python
# Quick test
python prediction_module.py

# In your code
from prediction_module import prediction
import numpy as np

# Load an image and preprocess (you implement this)
features = preprocess_image("test.jpg")

# Get prediction
face_count = prediction(features)
print(f"Faces detected: {face_count}")
```

---

## 📝 For Assignment Submission

1. **Add your details:**
   - Review the generated report

2. **Check results:**
   - `training_curves.png` - Visual performance
   - `training_results.json` - Detailed metrics
   - Should achieve >90% accuracy

3. **Submit these files:**
   - All `.py` files
   - `Assignment2_Report.md`
   - Dataset folders
   - Generated model files

---

## ⏱️ Time Estimates

- **Full automated setup:** ~5-10 minutes
- **With webcam capture:** +5-10 minutes  
- **With hyperparameter tuning:** +10-15 minutes
- **Total time:** ~15-30 minutes

---

## 💡 Pro Tips

1. **Use synthetic data first** to verify everything works
2. **Then add real images** for better performance
3. **Run tests** before submission: `python test_system.py`
4. **Keep virtual environment** for consistent results

---

## 🆘 Need Help?

1. Check error messages carefully
2. Run `python test_system.py` to diagnose issues
3. Review `README.md` for detailed documentation
4. Ensure Python 3.7+ is installed

---

Good luck! 🎉
