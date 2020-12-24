import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-sentiment-analysis',
  templateUrl: './sentiment-analysis.component.html',
  styleUrls: ['./sentiment-analysis.component.css']
})
export class SentimentAnalysisComponent {

  file:File = null!;
  isProcessing = false;
  isTrained = false;
  filePreview:string = '';
  sentimentSentence = '';

  constructor() {  }

  onSelectFile(event: Event) { // called each time file input changes
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) {
      return;
    }

    this.file = input.files[0];
    console.log(this.file);
    var reader = new FileReader();
    // reader.readAsDataURL(this.file); // read file as data url
    reader.readAsText(this.file);
    reader.onload = (event: Event) => { // called once readAsDataURL is completed
        this.filePreview = reader.result as string;;
    }
  }

  trainModel(){
    console.log("Training started...");
    this.isTrained = false;
    this.isProcessing = true;
    setTimeout(() => {
      console.log('hello');
      console.log("Training Completed...");
      this.isTrained = true;
      this.isProcessing = false;
    }, 5000);
  }

  predictSentiment(value: string): void {
    this.sentimentSentence = `${value}`;
    console.log('Sentiment Sentence: ' + this.sentimentSentence);
    this.isProcessing = true;
  }
}
