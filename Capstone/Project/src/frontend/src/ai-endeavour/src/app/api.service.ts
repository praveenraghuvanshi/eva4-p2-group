import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  BASE_URL = 'http://15.206.224.228';

  constructor(private httpClient: HttpClient) { }

  public upload(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);

    var response = this.httpClient.post<any>(this.BASE_URL + '/upload', formData);
    return response;
  }

  // Sentiment Analysis
  public train_sa(data_file: string): Observable<any> {
    console.log('Training data: ' + data_file);
    var trainUrl = this.BASE_URL + '/train/sa?data=' + data_file
    var response = this.httpClient.get<any>(trainUrl);
    return response;
 }

  public predict_sentiment(input: string, model: string, fields:string): Observable<any> {
    console.log('Input text: ' + input);
    console.log('Model file: ' + model);
    console.log('Text fields file: ' + fields);
    var inputData = {
       inputtext : input,
       model : model,
       textfields : fields
    }
    console.log(JSON.stringify(inputData))
    var response = this.httpClient.post<any>(this.BASE_URL + '/predict',
    JSON.stringify(inputData));
    return response;
 }

  // Image Classification
  public train_ic(base_directory: string): Observable<any> {
    console.log('Base directory: ' + base_directory);
    var trainUrl = this.BASE_URL + '/train/ic?data=' + base_directory;
    var response = this.httpClient.get<any>(trainUrl);
    return response;
  }

  public classify(image: string, model: string): Observable<any> {
    console.log('Image Src: ' + image);
    console.log('Model file: ' + model);

    var inputData = {
       image : image,
       model : model
    }
    console.log(JSON.stringify(inputData))
    var response = this.httpClient.post<any>(this.BASE_URL + '/classify',
    JSON.stringify(inputData));
    return response;
 }

 // Clear the dataset
  public clear(base_directory: string): Observable<any> {
    console.log('Base directory: ' + base_directory);
    var trainUrl = this.BASE_URL + '/clear?directory=' + base_directory;
    var response = this.httpClient.get<any>(trainUrl);
    return response;
  }

}
