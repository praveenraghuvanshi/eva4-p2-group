import { Component, OnInit } from '@angular/core';
import { ApiService } from '../api.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-translation',
  templateUrl: './translation.component.html',
  styleUrls: ['./translation.component.css']
})
export class TranslationComponent {
  public response: Observable<any>;
  germanSentence = '';
  translatedText = '';
  isProcessing = false;

  constructor(private apiService: ApiService){}

  translate(value: string): void {
    this.germanSentence = `${value}`;
    console.log('German: ' + this.germanSentence);
    this.isProcessing = true;
    this.apiService.translate(this.germanSentence).subscribe(data =>
      {
        console.log('Response: ' + data.output);
        this.translatedText = data.output;
        this.isProcessing = false;
      }); 
  }
}
