# Local deployment in WSL



# Prerequisites

1. Having WSL installed and ready
2. Clone the repo with ssh. Follow: https://docs.gitlab.com/topics/git/clone/#clone-with-ssh for more info. 
3. Go to the created directory in VScode and create a new branch. Follow tutorial: https://www.geeksforgeeks.org/git/how-to-create-a-new-branch-in-git/ for more info.

# 1. Installing WSL2 on windows 

Open windows powershell in adminstrator mode and then type: wsl --install. Once installed, restart system. For more details follow: https://learn.microsoft.com/de-de/windows/wsl/install

# 2. Install required libraries

1. Open a virtual environment (.venv) in the repository. 
2. Install the required libraries for both backend (requirements.txt) & frontend (requirements_frontend.txt) using the commands:<br>
    `pip install -r requirements.txt` and `pip install -r requirements_frontend.txt`

# 3. Install the required containers 

5 containers need to be installed: 
1. fastapi_container.sif (rebuild every time a new package is to be included in requirements.txt)
2. gradio.sif (rebuild if new package is to be included in requirements_frontend.txt)
3. redis.sif (for state management with fastapi)
4. ollama.sif for model serving (swap for vllm-openai_v0.9.1.sif in HPC)
5. milvus.v2.5.2.sif (for vectorDB backend)

1, 2 and 3 can be built using:<br>
`apptainer build {container_name} container_recipes/{container_recipe}` where container_recipe is either of
fastapi_recipe.def, gradio.def or redis.def.

For ollama and milvus containers, refer to _Apptainer_configs.docx_ file provided. 

# 4. Starting the webapplication and creating User IDs
1. Run the scripts for deployment of models, Redis, FastAPI backend and milvus vectorDB using the commands <br>
`sudo bash script_milvus.sh` and `bash script_backend.sh`
2. Update _script_frontend.sh_ to call the _start_CLI_ function in the last line (CLI is needed to create user ids and admin roles, rest can be done through gradio frontend)
3. Run `bash script_frontend.sh` to start the CLI and create a user profile with username, full name and password using `create` command (recommend to create names in same format as the ones used in HPC)
4. Assignment of user accounts to collections (existing or new) takes place using _assign_user_collection(user, collection)_ function in _helper_scripts/backend_admin.py_ which updates the _user_collection_db.json_ db.

# 5. Run gradio webapp
1. Update _script_frontend.sh_ to call the _start_gradio_ function in the last line. 
2. Run `bash script_frontend.sh` to start frontend (provided script_milvus and script_backend are already running). 
3. Go to http://localhost:8083/ to login using the created user id.


## Files not present in repo but need to be transferred to make app work
1. app/dev.env
2. _data_ folder with the pdfs to ingest 
3. frontend_gradio/utils/dev_frontend.env
