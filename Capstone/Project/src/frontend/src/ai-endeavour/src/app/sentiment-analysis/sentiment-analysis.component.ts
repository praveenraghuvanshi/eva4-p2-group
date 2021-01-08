import { Component } from '@angular/core';
import { ApiService } from '../api.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-sentiment-analysis',
  templateUrl: './sentiment-analysis.component.html',
  styleUrls: ['./sentiment-analysis.component.css']
})
export class SentimentAnalysisComponent {

  public response: Observable<any>;
  file:File = null!;
  uploading = false;
  training = false;
  hasTrained = false;
  predicting = false;
  filePreview:string = '';
  sentimentSentence = '';
  uploadedFile:string = ''
  predictedSentiment = ''
  model = '';
  textfields = ''
  accuracy = 0
  public csvRecords: any[] = [];
  data: any;
  headers: string[] = [];

  constructor(private apiService: ApiService) {  }

  onSelectFile(event: Event) { // called each time file input changes
    const input = event.target as HTMLInputElement;
    if (!input.files.length) {
      return;
    }

    this.file = input.files[0];
    console.log(this.file);
    this.uploading = true;
    var reader = new FileReader();
    // reader.readAsDataURL(this.file); // read file as data url
    reader.readAsText(this.file);
    reader.onload = (event: Event) => { // called once readAsDataURL is completed
      this.filePreview = reader.result as string;
      // Get 10 CSV records
      let csvData = reader.result;
      const results = this.csvToJSON(csvData as string);
      this.data = results.data.slice(0, 10);
      this.headers = results.headers;
    }
    this.apiService.upload(this.file).subscribe(data =>
      {
        console.log('Response: ' + JSON.stringify(data));
        this.uploadedFile = data.file;
        this.uploading = false;
      },
      error => { //Error callback
        console.error('error caught in component\nError Details: ' + JSON.stringify(error))
        this.uploading = false;
        alert("An error occured while processing the request, Please retry the operation!!!");
      });
  }

  trainModel(){
    console.log("Training started...");
    this.training = true;
    this.apiService.train_sa(this.uploadedFile).subscribe(data =>
      {
        console.log(JSON.stringify(data));
        this.training = false;
        this.hasTrained = true;
        this.model = data.model;
        this.textfields = data.text_fields_file;
        this.accuracy = Math.round(data.test_acc * 100 * 100) / 100;
        console.log("Training completed, Accuracy: " + this.accuracy);
      },
      error => { //Error callback
        console.error('error caught in component\nError Details:' + JSON.stringify(error));
        this.training = false;
        this.hasTrained = true;
        alert("An error occured while processing the request, Please retry the operation!!!");
      });
  }

  predictSentiment(value: string): void {
    this.sentimentSentence = `${value}`;
    console.log('Sentiment Sentence: ' + this.sentimentSentence);
    this.predicting = true;
    this.apiService.predict_sentiment(this.sentimentSentence, this.model, this.textfields).subscribe(data =>
      {
        console.log(JSON.stringify(data));

        this.predictedSentiment = data.prediction.toLowerCase();
        console.log("Predicted Sentiment: " + this.predictedSentiment);
        this.predicting = false;
      },
      error => { //Error callback
        console.error('error caught in component\nError Details:' + JSON.stringify(error));
        this.predicting = false;
        alert("An error occured while processing the request, Please retry the operation!!!");
      });
  }

    // CSV is assumed to have headers as well
    // https://stackblitz.com/edit/angular-k162aa?file=src%2Fapp%2Fapp.component.html
  csvToJSON(csv: string) {

      const lines: string[] = csv
        // escape everything inside quotes to NOT remove the comma there
        .replace(/"(.*?)"/gm, (item) => encodeURIComponent(item))
        .split('\n');

      // separate the headers from the other lines and split them
      const headers: string[] = lines.shift().split(',');

      // should contain all CSV lines parsed for the html table
      const data: any[] = lines.map((lineString, index) => {
        const lineObj = {};

        const lineValues = lineString.split(',');

        headers.forEach((valueName, index) => {
          // remove trailing spaces and quotes
          lineObj[valueName] = lineValues[index]
            // handle quotes
            .replace(/%22(.*?)%22/gm, (item) => decodeURIComponent(item))
            .trim();
        })

        return lineObj; // return lineObj for objects.
      });

      return { data, headers };
    }
}
