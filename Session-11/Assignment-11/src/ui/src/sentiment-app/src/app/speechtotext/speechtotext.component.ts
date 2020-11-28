import { Component, OnInit } from '@angular/core';
import { ApiService } from '../api.service';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-speechtotext',
  templateUrl: './speechtotext.component.html',
  styleUrls: ['./speechtotext.component.css']
})
export class SpeechtotextComponent {

  url = '';
  convertedText = ''
  file: any
  isProcessing = false;
  
  public response: Observable<any>;

  constructor(private apiService: ApiService) { }
  
  onSelectFile(event) { // called each time file input changes
      if (event.target.files && event.target.files[0]) {
        console.log(event.target.files[0]);
        this.file = event.target.files[0];
        var reader = new FileReader();
        reader.readAsDataURL(event.target.files[0]); // read file as data url
        reader.onload = (event: Event) => { // called once readAsDataURL is completed
          this.url = reader.result as string;          
        }
      }
  }

  speechToTextCaption(){
    this.isProcessing = true;
    this.apiService.generateCaption(this.file).subscribe(data =>
      {
        console.log('Response: ' + data.output);
        this.convertedText = data.output;
        this.isProcessing = false;
      },
      error => { //Error callback
        console.error('error caught in component')
        // this.errorMessage = error;
        this.isProcessing = false;
        alert("An error occured while processing the request, Please retry the operation!!!");
      });
  }
}
