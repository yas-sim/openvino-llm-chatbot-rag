from huggingface_hub import login, whoami

try:
    whoami()
    print('Authorization token already provided')
except OSError:
    login()
