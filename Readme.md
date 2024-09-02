# Call Result Prediction

## Getting Started

Follow the steps below to set up and run the project.

### 1. Clone the Repository

Clone this repository using the following command:

```bash
git clone <repository_url>
```

### 2. Activate virtual environment

- For windows terminal `./menv/Scripts/activate `
- For Linux terminal ` source menv/Scripts/activate`

### 3. Install Dependencies

`pip install -r requirements.txt`

### 4. Unzip data file

`unzip ./app/data/data.zip -d ./app/data/`

### 5. Add model

- Add model from [model-tokenizer](https://drive.google.com/file/d/1UdCpLJJdTnSY3cRReDMnyHWjvEmk9v0I/view?usp=sharing) to folder ```./app/my_finetuned_model/```

- unzip model_tokenizer.zip

## How to run Service

### Start services

`uvicorn app.main:app --host 0.0.0.0 --port 8080`

## How to retrain model

### Retrain model

`python app/training.py`
