# EVA-4 Phase-2 : Assignment - 5

### Team Members

- Praveen Raghuvanshi (praveenraghuvanshi@gmail.com)
- Tusharkanta Biswal (Tusharkanta_biswal@stragure.com)
- Suman Kanukollu (sumankanukollu@gmail.com)
- Shubham Kothawade (kothawadeshub@gmail.com)

## Resources

- Webpage: https://eva4p2bucket1.s3.ap-south-1.amazonaws.com/src/index.html
- Github: https://github.com/praveenraghuvanshi/eva4-p2-group/tree/master/Session-5/Assignment-5
- Source: https://github.com/praveenraghuvanshi/eva4-p2-group/tree/master/Session-5/Assignment-5/src
- Notebooks: https://github.com/praveenraghuvanshi/eva4-p2-group/blob/master/Session-5/Assignment-5/src/eva4p2s5.ipynb
- [Paper Review](PaperReview.md)

## Result

<img src=".\assets\hpe-result.png" alt="Human Pose Estimation" style="zoom:80%;" />

## Source

**[index.html](src/index.html)**

**[upload.js](src/js/upload.js)**

**[handler.py](src/serverless/handler.py)**

**[serverless.yml](src/serverless/serverless.yml)**

**SLS Deploy Log:**

```powershell
Serverless: Docker Image: lambci/lambda:build-python3.8
Serverless: Using download cache directory /home/skanukollu/suman/EVA4_P2/suman_cnn/sessions/s5_HPE/service/cache/downloadCacheslspyc
Serverless: Running docker run --rm -v /home/skanukollu/suman/EVA4_P2/suman_cnn/sessions/s5_HPE/service/cache/52a08b26640952b9c3fa49694982cd3972ad932c39a7635038b97f290fd39eea_slspyc\:/var/task\:z -v /home/skanukollu/suman/EVA4_P2/suman_cnn/sessions/s5_HPE/service/cache/downloadCacheslspyc\:/var/useDownloadCache\:z lambci/lambda\:build-python3.8 /bin/sh -c 'chown -R 0\\:0 /var/useDownloadCache && python3.8 -m pip install -t /var/task/ -r /var/task/requirements.txt --cache-dir /var/useDownloadCache && chown -R 1002\\:1002 /var/task && chown -R 1002\\:1002 /var/useDownloadCache'...
Serverless: Zipping required Python packages...
Serverless: Packaging service...
Serverless: Excluding development dependencies...
Serverless: Removing Python requirements helper...
Serverless: Injecting required Python packages to package...
Serverless: WARNING: Function main_handler has timeout of 60 seconds, however, it's attached to API Gateway so it's automatically limited to 30 seconds.
Serverless: Uploading CloudFormation file to S3...
Serverless: Uploading artifacts...
Serverless: Uploading service s5hpe.zip file to S3 (110.63 MB)...
Serverless: Validating template...
Serverless: Updating Stack...
Serverless: Checking Stack update progress...
..............
Serverless: Stack update finished...
Service Information
service: s5hpe
stage: dev
region: ap-south-1
stack: s5hpe-dev
resources: 12
api keys:
  None
endpoints:
  POST - https://6lm64acf6k.execute-api.ap-south-1.amazonaws.com/dev/hpe
functions:
  main_handler: s5hpe-dev-main_handler
layers:
  None
Serverless: Removing old service artifacts from S3...

```
