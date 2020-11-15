# EVA-4 Phase-2 : Assignment - 11

### Team Members

- Praveen Raghuvanshi (praveenraghuvanshi@gmail.com)
- Tusharkanta Biswal (Tusharkanta_biswal@stragure.com)
- Suman Kanukollu (sumankanukollu@gmail.com)

## Resources

- Webpage: https://hcrjlpjoo0.execute-api.ap-south-1.amazonaws.com/production/translationComponent
- Github: https://github.com/praveenraghuvanshi/eva4-p2-group/tree/master/Session-11/Assignment-11
- Source: https://github.com/praveenraghuvanshi/eva4-p2-group/tree/master/Session-11/Assignment-11/src
- Notebooks: https://github.com/praveenraghuvanshi/eva4-p2-group/blob/master/Session-11/Assignment-11/src/ev4p2s11.ipynb

## Result

### Output

**Input text:** mein vater hörte sich auf seinem kleinen , grauen radio die der bbc an .

**Translated Text:** my father stopped on his little bit gray , gray radio the bbc bbc .

<img src="assets\translation-result-new-ui.png" alt="Translated Result" style="zoom:80%;" />

### Performance

<img src="assets\attention-model-performance.png" alt="Attention Model Performance" style="zoom:80%;" />

### Translation

```javascript
Example #2
Src :  mein vater hörte sich auf seinem kleinen , grauen radio die <unk> der bbc an .
Trg :  my father was listening to bbc news on his small , gray radio .
Pred:  my father was focused on his little , gray radio the <unk> of the bbc .
```

## Source

**[index.html](src/index.html)**

**[upload.js](src/js/upload.js)**

**[handler.py](src/serverless/handler.py)**

**[serverless.yml](src/serverless/serverless.yml)**

