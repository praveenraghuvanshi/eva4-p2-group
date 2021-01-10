import os
import boto3
import urllib
from pathlib import Path
from werkzeug.utils import secure_filename

CSV_EXTENSION = ".csv"
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif'}
BUCKET_NAME = "aiendeavourbkt"

### Application Level Initialization
s3 = boto3.resource(u's3')


def upload_file(file, uploadToRemote = False):
    filePath = get_file_path(file)
    print(filePath)
    file.save(filePath)
    assert(os.path.exists(filePath))

    #if uploadToRemote:
    #    filePath = upload_to_s3(filePath, filePath)

    return str(filePath)

def get_file_path(file):
    filename, file_extension = os.path.splitext(file.filename)
    if file_extension == CSV_EXTENSION:
        return file.filename
    elif file_extension in IMAGE_EXTENSIONS:
        filePathParts = len(Path(file.filename).parts) == 3 
        if filePathParts and '.' in file.filename:
            osFilePath = Path.cwd().joinpath(file.filename)
            if not osFilePath.exists():
                osFilePath.parent.mkdir(exist_ok=True,parents=True)
            return str(osFilePath)
        else:
            raise RuntimeError("Unsupported Directory structure. Supported structure is MainDirectory/ClassDirecotry/Image")
    else:
        raise RuntimeError("Unsupported File type")

### AWS Specific Functions
def upload_to_s3(localFile, remoteFile):
    s3.Bucket(BUCKET_NAME).upload_file(localFile, remoteFile)
    uploadedFileUrl = "https://s3-%s.amazonaws.com/%s/%s" % (
        "ap-south-1",
        BUCKET_NAME,
        urllib.parse.quote(remoteFile, safe="~()*!.'"),
    )
    print('S3 uploaded file Url is ' + remoteFile)
    print("S3 url is " + uploadedFileUrl)
    return uploadedFileUrl

def download_from_s3(local_file, remote_file):
    s3.Bucket(BUCKET_NAME).download_file(remote_file, local_file)
    return local_file