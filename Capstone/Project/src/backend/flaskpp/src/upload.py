import os
from pathlib import Path
from werkzeug.utils import secure_filename

CSV_EXTENSION = ".csv"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif'}

def upload_file(file):
    filePath = get_file_path(file)
    print(filePath)
    file.save(filePath)
    assert(os.path.exists(filePath))
    return str(filePath)

def get_file_path(file):
    filename, file_extension = os.path.splitext(file.filename)
    if file_extension == CSV_EXTENSION:
        return file.filename
    elif file_extension in IMAGE_EXTENSIONS:
        filePathParts = len(Path(file.filename).parts) == 3 
        if filePathParts and '.' in file.filename:
            filepath = Path.cwd().joinpath(file.filename)
            if not filepath.exists():
                filepath.parent.mkdir(exist_ok=True,parents=True)
            return filepath
        else:
            raise RuntimeError("Unsupported Directory structure. Supported structure is MainDirectory/ClassDirecotry/Image")
    else:
        raise RuntimeError("Unsupported File type")
