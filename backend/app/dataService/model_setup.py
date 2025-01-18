import os
import sys
from huggingface_hub import hf_hub_download
from app.dataService.globalVariable import LLAMA_MODEL_PATH, LLAVA_MODEL_PATH, llama_dir, llava_dir

def download_llama():
    """Download Llama model"""
    if not os.path.exists(LLAMA_MODEL_PATH):
        print("Downloading Llama model...")
        try:
            hf_hub_download(
                repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
                filename="llama-2-7b-chat.Q4_K_M.gguf",
                local_dir=llama_dir,
                local_dir_use_symlinks=False
            )
            # Rename to standard name
            os.rename(
                os.path.join(llama_dir, "llama-2-7b-chat.Q4_K_M.gguf"),
                LLAMA_MODEL_PATH
            )
            print(f"Llama model downloaded to: {LLAMA_MODEL_PATH}")
        except Exception as e:
            print(f"Error downloading Llama model: {e}")
            sys.exit(1)
    else:
        print("Llama model already exists")

def download_llava():
    """Download LLaVA model"""
    if not os.path.exists(LLAVA_MODEL_PATH):
        print("Downloading LLaVA model...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="liuhaotian/llava-v1.5-13b",
                local_dir=LLAVA_MODEL_PATH,
                local_dir_use_symlinks=False
            )
            print(f"LLaVA model downloaded to: {LLAVA_MODEL_PATH}")
        except Exception as e:
            print(f"Error downloading LLaVA model: {e}")
            sys.exit(1)
    else:
        print("LLaVA model already exists")

def setup_models():
    """Setup all models"""
    print("Setting up models...")
    download_llama()
    download_llava()
    print("Model setup complete!")

if __name__ == "__main__":
    setup_models()