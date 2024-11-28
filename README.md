# 📝 SQL Helper: Natural Language ↔ SQL Interactive System

An AI system that provides:
- Natural Language → SQL: Convert user questions to SQL queries
- SQL → Natural Language: Explain query results in plain language
- Interactive Refinement: Multi-turn conversation to improve query accuracy

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
- Backend Server (GPU)
  - NVIDIA L4 GPU minimum
  - VRAM: 23GB+ (Tested: 23034MiB)
  - CUDA Version: 12.2
  - Driver Version: 535.183.01+
  - Purpose: LLM processing & SQL generation

- Frontend Server (CPU)
  - Standard CPU instance
  - Memory: 8GB+ recommended
  - Purpose: Web interface & user interactions

### Required Open Ports
- Port 8501: Streamlit web interface access
- Port 8000: Backend FastAPI server access 
- Port 3306: MySQL database connection



# Installation and Setup
## Backend Setup Guide
### GPU Server Initial Setup

1. Install CUDA and NVIDIA drivers:
```bash
cd sql-helper/backend/GPUsetting
chmod +x cuda_install.sh
./cuda_install.sh
# System will reboot
```
Note: Server requires reboot after CUDA installation. Ensure all commands are executed in order.

2. After reboot, install PyEnv dependencies:
```bash
cd sql-helper/backend/GPUsetting
chmod +x pyenv_dependencies.sh
./pyenv_dependencies.sh
```
3. Setup PyEnv:
```bash
chmod +x pyenv_setup.sh
./pyenv_setup.sh
```


4. Create Python environment:
```bash
chmod +x pyenv_virtualenv.sh
./pyenv_virtualenv.sh
# Enter Python version: 3.11.8
# Enter environment name: backend
```

5. Verify GPU setup:
```bash
nvidia-smi
# Should show NVIDIA L4 GPU info
```

### Backend Application Setup
1. Setup backend application:
```bash
cd sql-helper/backend
chmod +x backend_env_setup.sh
./backend_env_setup.sh
```

2. Configure environment variables:
Create .env file in project root:
```
OPENAI_API_KEY="your-api-key"
URL="your-mysql-database-url"
HUGGINGFACE_TOKEN='your-huggingface-token"
```

3. Start backend:
```
python main.py
```

## Frontend Setup Guide

0. Clone and start application:
```bash
git clone https://github.com/100suping/sql-helper.git
```

1. Run environment setup script:
```bash
cd sql-helper/frontend
chmod +x frontend_env_setup.sh
./frontend_env_setup.sh
```

2. Open new terminal and activate environment:
```bash
pyenv activate frontend
```

3. Configure environment variables:
Create .env file in project fronted folder:
```
BACKEND_HOST="your backend server ip"
DB_HOST=localhost
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=your_database_name
```

## Frontend Application run

0. Start application:
```   
cd sql-helper/frontend
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
sql-helper/
├── frontend/
│   ├── app.py                # Streamlit interface
│   ├── requirements.txt      # Frontend dependencies
│   ├── README.md            # Frontend docs
│   └── frontend_env_setup.sh # Frontend setup script
│   └── .env      # Frontend variables for front
├── backend/
│   ├── GPUsetting/          # GPU/CUDA setup scripts
│   │   ├── cuda_install.sh
│   │   ├── pyenv_dependencies.sh
│   │   ├── pyenv_setup.sh
│   │   └── pyenv_virtualenv.sh
│   ├── langgraph_/          # Core backend logic
│   │   ├── init.py
│   │   ├── faiss_init.py
│   │   ├── graph.py
│   │   ├── node.py
│   │   ├── task.py
│   │   └── utils.py
│   ├── prompts/             # LLM prompts
│   │   ├── additional_question/
│   │   ├── general_conversation/
│   │   ├── query_creation/
│   │   ├── question_analysis/
│   │   ├── question_evaluation/
│   │   ├── question_refinement/
│   │   ├── sql_conversation/
│   │   └── table_selection/
│   ├── backend_env_setup.sh # Backend setup script
│   ├── main.py             # Backend entry point
│   ├── README.md           # Backend docs
│   └── requirements.txt    # Backend dependencies
│   └── .env      # Environment variables for backend
├── README.md              # Project documentation
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

- [Sessac101/sql-helper-tone-QA](https://huggingface.co/datasets/Sessac101/sql-helper-tone-QA)
- Modified BIRD dataset with Korean translations
 - Cleaned and merged schema
 - Added Korean Q&A pairs
 - Available files:
   - `merged_cleaned.json`: Cleaned dataset
   - `merged_cleaned_addschema.json`: Dataset with schema
   - `모범 QA - 시트1.csv`: 50 curated Q&A pairs

- [won75/text_to_sql_ko](https://huggingface.co/datasets/won75/text_to_sql_ko)
 - Korean text-to-SQL dataset
 - Used for enhancing Korean language support
 - Based on Spider dataset

Both datasets were used for fine-tuning our model for improved Korean SQL generation.
