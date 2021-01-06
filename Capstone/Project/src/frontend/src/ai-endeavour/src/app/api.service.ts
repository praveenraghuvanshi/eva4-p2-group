import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  constructor(private httpClient: HttpClient) { }

  public upload(file: File): Observable<any> {
    console.log("File to be uploaded: " + file.name);
    const formData = new FormData();
    formData.append('file', file);

    var response = this.httpClient.post<any>('http://ec2-65-0-116-36.ap-south-1.compute.amazonaws.com/upload', formData);
    return response;
  }

  public train(data_file: string): Observable<any> {
    console.log('Training data: ' + data_file);
    var trainUrl = 'http://ec2-65-0-116-36.ap-south-1.compute.amazonaws.com/train?data=' + data_file
    var response = this.httpClient.get<any>(trainUrl);
    return response;
 }

  public predict(input: string, model: string, fields:string): Observable<any> {
    console.log('Input text: ' + input);
    console.log('Model file: ' + model);
    console.log('Text fields file: ' + fields);
    var inputData = {
       inputtext : input,
       model : model,
       textfields : fields
    }
    console.log(JSON.stringify(inputData))
    var response = this.httpClient.post<any>('http://ec2-65-0-116-36.ap-south-1.compute.amazonaws.com/predict',
    JSON.stringify(inputData));
    return response;
 }
}
