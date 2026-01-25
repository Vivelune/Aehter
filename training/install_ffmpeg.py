import subprocess
import sys

def install_ffmpeg():
    print(f"Starting FFMPEG installation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--ffmpeg-python"])
        print("Installed ffmpeg-python successfully")
    except subprocess.CalledProcessError as e:
        print("Failed to install ffmpeg-python via pip")

    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-0", "/tmp/ffmpeg.tar.xz"
        ])
        
        subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg" , "-type" , "f"],
            capture_output=True,
            text=True
        )
        ffmpeg_path = result.stdout.strip()
        
        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])
        subprocess.check_call(["chmod", '+x', "/usr/local/bin/ffmpeg"])
        print("Installed static FFmpeg binary successfully ")
    except Exception as e:
        print(f"Failed to install static FFMPEG : {e}")
        
    try: 
        result = subprocess.run(["ffmpeg" "-version"], capture_output=True, text=True, check=True)
        print("FFMPEG version:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFMPEG installation verificationf failed")
        return False

        
        


    