`
# 🐍 Python Project Setup with venv

dataset: ```bash https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd```
---

## 📁 Project Structure

```
my-python-project/
│
├── venv/               # Virtual environment directory
├── templates/          # Your html file
├── app.py              # Your main Python file
├── train_model.py      # Your Training model
├── requirements.txt    # List of dependencies
└── README.md           # This file
```

---

## ✅ Requirements

- Python 3.x installed
- Git (optional)
- recomended 3.10.0 
---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/rustam76/drowsiness-detection.git
cd drowsiness-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

- On **Windows**:

```bash
venv\Scripts\activate
```

- On **macOS/Linux**:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

---

## ❎ Deactivate Virtual Environment

To deactivate the environment:

```bash
deactivate
```

---

## 📝 Notes

- Make sure `requirements.txt` is updated by running:

```bash
pip freeze > requirements.txt
```

- Don't forget to include `venv/` in `.gitignore` to avoid pushing it to Git.

```
# .gitignore
venv/
```

---

## 📫 Contact

For any questions, feel free to reach out!
