# 📝 SQL Helper: Natural Language ↔ SQL Interactive System

An AI system that provides:
- Natural Language → SQL: Convert user questions to SQL queries
- SQL → Natural Language: Explain query results in plain language
- Interactive Refinement: Multi-turn conversation to improve query accuracy


## Demo
![demo](https://github.com/user-attachments/assets/5d74f962-fcc3-4f88-b151-f6128c8b5ae9)

## Required Files
### Database Dump
Download the MySQL dump file from [Release v1.0](https://github.com/100suping/sql-helper/releases/tag/mysql-dump)
Import database:
```bash
mysql -u your_user -p your_database < dump.sql
```

### Core Components
- Frontend: Streamlit web interface
- Backend: LangGraph query generation
- RAG: Database schema-aware system using FAISS vector store
  - Indexes table structures and relationships
  - Uses metadata for precise SQL generation
  - Enables context-aware query refinement
- Database: MySQL integration

### System Architecture
- Model Server (GPU)
  - NVIDIA L4 GPU minimum
  - VRAM: 23GB+ (Tested: 23034MiB)
  - CUDA Version: 12.2
  - Driver Version: 535.183.01+
  - Purpose: LLM processing & SQL generation

- Backend Server (CPU)
  - Standard CPU instance
  - Memory: 8GB+ recommended
  - Purpose: Query processing & orchestration

- Frontend Server (CPU)
  - Standard CPU instance
  - Memory: 8GB+ recommended
  - Purpose: Web interface & user interactions

### Required Open Ports
- Port 8501: Streamlit web interface access
- Port 8000: Backend FastAPI server access 
- Port 3306: MySQL database connection



# Installation and Setup


0. Clone and start application:
```bash
git clone https://github.com/100suping/sql-helper.git
```


## Model Setup Guide
### GPU Server Initial Setup

1. Install CUDA and NVIDIA drivers:
```bash
cd sql-helper/model/GPUsetting
chmod +x cuda_install.sh
bash cuda_install.sh
# System will reboot
```
Note: Server requires reboot after CUDA installation. Ensure all commands are executed in order.

2. After reboot, install PyEnv dependencies:
```bash
cd sql-helper/model/GPUsetting
chmod +x pyenv_dependencies.sh
bash pyenv_dependencies.sh
```
3. Setup PyEnv:
```bash
cd sql-helper/model/GPUsetting
chmod +x pyenv_setup.sh
bash pyenv_setup.sh
```

4. Create Python environment:
```bash
cd sql-helper/model/GPUsetting
chmod +x pyenv_virtualenv.sh
bash pyenv_virtualenv.sh
# Enter Python version: 3.11.8
# Enter environment name: model
```

5. run your python environment and 2 bash files
```bash
# pyenv activate model
cd sql-helper/model
chmod +x model_server_setting.sh
bash model_server_setting.sh
# model cache would be located in sql-helper/.cache
```

6. Verify GPU setup:
```bash
nvidia-smi
# Should show NVIDIA L4 GPU info
```

7. Configure environment variables:
Create .env file in project root:
```bash
OPENAI_API_KEY="your-api-key"
HUGGINGFACE_TOKEN="your-huggingface-token"
```

8.Start model server:
```bash
bash main.sh
```

### Backend Setup
1. Setup backend application:
```bash
cd sql-helper/backend
chmod +x backend_env_setup.sh
bash backend_env_setup.sh
```

2. run your python environment
```bash
# if you want to run your pyenv enviroment to activate model
cd sql-helper/backend
pip install -r requirements.txt
```

3. Configure environment variables:
Create .env file in project root:
```bash
URL="your-mysql-database-url"
MODEL_HOST="your-model-server-ip"
OPENAI_API_KEY="your-api-key"
```

3. Start backend:
```bash
python main.py
```

## Frontend Setup Guide

1. Setup frontend application:
```bash
cd sql-helper/frontend
chmod +x frontend_env_setup.sh
bash frontend_env_setup.sh
```

2. Open new terminal and activate environment:
```bash
# if you want to run your pyenv enviroment to activate model
pip install -r requirements.txt
```

3. Configure environment variables:
Create .env file in project fronted folder:
```bash
DB_HOST="your mysql DB server ip"
DB_USER="your root account"
DB_PASSWORD="your root account's password"
DB_NAME=user_db
BACKEND_HOST="your backend server ip"
```

## Frontend Application run

0. Start application:
```bash   
streamlit run app.py
```

## Project Structure
```
sql-helper/
├── frontend/
│   ├── app.py                # Streamlit interface
│   ├── requirements.txt      # Frontend dependencies
│   ├── README.md            # Frontend documentation
│   ├── frontend_env_setup.sh # Frontend setup script
│   └── .env                 # Frontend environment variables
├── backend/
│   ├── langgraph_/          # Core backend logic
│   │   ├── __init__.py
│   │   ├── faiss_init.py
│   │   ├── graph.py
│   │   ├── node.py
│   │   ├── task.py
│   │   └── utils.py
│   ├── prompts/             # LLM prompts
│   ├── backend_env_setup.sh # Backend setup script
│   ├── main.py             # Backend entry point
│   ├── README.md           # Backend documentation
│   ├── requirements.txt    # Backend dependencies
│   └── .env               # Backend environment variables
├── model/
│   ├── GPUsetting/         # GPU/CUDA setup scripts
│   │   ├── cuda_install.sh
│   │   ├── pyenv_dependencies.sh
│   │   ├── pyenv_setup.sh
│   │   └── pyenv_virtualenv.sh
│   ├── main.py            # Model server entry point
│   ├── main.sh            # Model server startup script
│   ├── model_server_setting.sh # Model server configuration
│   ├── utils.py           # Utility functions
│   ├── requirements.txt   # Model dependencies
│   ├── README.md         # Model documentation
│   └── .env             # Model environment variables
├── README.md            # Project documentation
└── .gitignore

Note: `.env` file should be placed in project root and backend directory needs access to it for database and API configurations.

```

## LLM Models Used
### Main Model
- [100suping/Qwen2.5-Coder-34B-Instruct-kosql-adapter](https://huggingface.co/100suping/Qwen2.5-Coder-34B-Instruct-kosql-adapter)
 - Fine-tuned for SQL generation and natural language interaction
 - Optimized for Korean language support
 - Access requires Hugging Face token

## Training Datasets

- [won75/text_to_sql_ko](https://huggingface.co/datasets/won75/text_to_sql_ko)
 - Korean text-to-SQL dataset
 - Used for enhancing Korean language support
 - Based on Spider dataset

- [Sessac101/sql-helper-tone-QA](https://huggingface.co/datasets/Sessac101/sql-helper-tone-QA)
- Modified BIRD dataset with Korean translations
 - Cleaned and merged schema
 - Added Korean Q&A pairs
 - Available files:
   - `merged_cleaned.json`: Cleaned dataset
   - `merged_cleaned_addschema.json`: Dataset with schema
   - `모범 QA - 시트1.csv`: 50 curated Q&A pairs

Both datasets were used for fine-tuning our model for improved Korean SQL generation.
