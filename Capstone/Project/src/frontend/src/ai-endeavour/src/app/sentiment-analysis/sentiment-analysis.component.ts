import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-sentiment-analysis',
  templateUrl: './sentiment-analysis.component.html',
  styleUrls: ['./sentiment-analysis.component.css']
})
export class SentimentAnalysisComponent {

  constructor() {
    console.log("Inside SentimentAnalysis ctor()");
  }
}
