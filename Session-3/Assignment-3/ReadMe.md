# EVA-4 Phase-2 : Assignment - 3

### Team Members

- Praveen Raghuvanshi (praveenraghuvanshi@gmail.com)
- Tusharkanta Biswal (Tusharkanta_biswal@stragure.com)
- Suman Kanukollu (sumankanukollu@gmail.com)
- Shubham Kothawade (kothawadeshub@gmail.com)

## Resources

- Webpage: https://eva4p2bucket1.s3.ap-south-1.amazonaws.com/src/index.html
- Github: https://github.com/praveenraghuvanshi/eva4-p2-group/tree/master/Session-3/Assignment-3
- Source: https://github.com/praveenraghuvanshi/eva4-p2-group/tree/master/Session-3/Assignment-3/src
- Notebooks: https://github.com/praveenraghuvanshi/eva4-p2-group/blob/master/Session-3/Assignment-3/src/eva4p2s3.ipynb

## Result

<img src=".\assets\face-alignment-result.png" alt="Face alignment" style="zoom:80%;" />

## Source

**[index.html](src/index.html)**

**[upload.js](src/js/upload.js)**

**[handler.py](src/serverless/handler.py)**

**[serverless.yml](src/serverless/serverless.yml)**

## Steps to create new serverless API

- Start powershell in admin mode
- Set execution policy 
  - Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
- sls create --template aws-python3 --name <service-name>
- serverless plugin install --name serverless-python-requirements
- npm run deploy
- Grant permission to docker for file sharing
- Update API gateway settings : multipart/form-data and */*