# EVA-4 Phase-2 : Assignment - 6

### Team Members

- Praveen Raghuvanshi (praveenraghuvanshi@gmail.com)
- Tusharkanta Biswal (Tusharkanta_biswal@stragure.com)
- Suman Kanukollu (sumankanukollu@gmail.com)
- Shubham Kothawade (kothawadeshub@gmail.com)

## Resources

- Webpage: https://eva4p2bucket1.s3.ap-south-1.amazonaws.com/src/index.html
- Github: https://github.com/praveenraghuvanshi/eva4-p2-group/tree/master/Session-6/Assignment-6
- Source: https://github.com/praveenraghuvanshi/eva4-p2-group/tree/master/Session-6/Assignment-6/src
- Notebooks: https://github.com/praveenraghuvanshi/eva4-p2-group/blob/master/Session-6/Assignment-6/src/eva4p2s6.ipynb

## Result

<img src=".\assets\gans-real-fake.png" alt="GAN - Real Vs Fake" style="zoom:80%;" />



<img src=".\assets\gans-cars.gif" alt="Gans on cars" style="zoom:80%;" />





## Source

**[index.html](src/index.html)**

**[upload.js](src/js/upload.js)**

**[handler.py](src/serverless/handler.py)**

**[serverless.yml](src/serverless/serverless.yml)**

**SLS Deploy Log:**

```powershell
suman/EVA4_P2/suman_cnn/sessions/s6_GAN_Car_real_fake/service$ npm run deploy
> s5hpe@0.1.0 deploy /home/skanukollu/suman/EVA4_P2/suman_cnn/sessions/s6_GAN_Car_real_fake/service
> serverless deploy
Serverless: Adding Python requirements helper...
Serverless: Generated requirements from /home/skanukollu/suman/EVA4_P2/suman_cnn/sessions/s6_GAN_Car_real_fake/service/requirements.txt in /home/skanukollu/suman/EVA4_P2/suman_cnn/sessions/s6_GAN_Car_real_fake/service/.serverless/requirements.txt...
Serverless: Installing requirements from /home/skanukollu/suman/EVA4_P2/suman_cnn/sessions/s6_GAN_Car_real_fake/service/cache/065286d65030f853e76cf0d82e82c9f58988768f8bd835d4d06751b8de94307c_slspyc/requirements.txt ...
Serverless: Docker Image: lambci/lambda:build-python3.8
Serverless: Using download cache directory /home/skanukollu/suman/EVA4_P2/suman_cnn/sessions/s6_GAN_Car_real_fake/service/cache/downloadCacheslspyc
Serverless: Running docker run --rm -v /home/skanukollu/suman/EVA4_P2/suman_cnn/sessions/s6_GAN_Car_real_fake/service/cache/065286d65030f853e76cf0d82e82c9f58988768f8bd835d4d06751b8de94307c_slspyc\:/var/task\:z -v /home/skanukollu/suman/EVA4_P2/suman_cnn/sessions/s6_GAN_Car_real_fake/service/cache/downloadCacheslspyc\:/var/useDownloadCache\:z lambci/lambda\:build-python3.8 /bin/sh -c 'chown -R 0\\:0 /var/useDownloadCache && python3.8 -m pip install -t /var/task/ -r /var/task/requirements.txt --cache-dir /var/useDownloadCache && chown -R 1002\\:1002 /var/task && chown -R 1002\\:1002 /var/useDownloadCache'...
Serverless: Zipping required Python packages...
Serverless: Packaging service...
Serverless: Excluding development dependencies...
Serverless: Removing Python requirements helper...
Serverless: Injecting required Python packages to package...
Serverless: WARNING: Function main_handler has timeout of 60 seconds, however, it's attached to API Gateway so it's automatically limited to 30 seconds.
Serverless: Creating Stack...
Serverless: Checking Stack create progress...
........
Serverless: Stack create finished...
Serverless: Uploading CloudFormation file to S3...
Serverless: Uploading artifacts...
Serverless: Uploading service s6ganrealfake.zip file to S3 (147.5 MB)...
Serverless: Validating template...
Serverless: Updating Stack...
Serverless: Checking Stack update progress...
.................................
Serverless: Stack update finished...
Service Information
service: s6ganrealfake
stage: dev
region: ap-south-1
stack: s6ganrealfake-dev
resources: 12
api keys:
  None
endpoints:
  POST - https://pjoo5furs2.execute-api.ap-south-1.amazonaws.com/dev/hpe
functions:
  main_handler: s6ganrealfake-dev-main_handler
layers:
  None

   ╭────────────────────────────────────────────────────────────────╮
   │                                                                │
   │      New minor version of npm available! 6.13.4 -> 6.14.8      │
   │   Changelog: https://github.com/npm/cli/releases/tag/v6.14.8   │
   │               Run npm install -g npm to update!                │
   │                                                                │
   ╰────────────────────────────────────────────────────────────────╯
```
