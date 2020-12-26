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
    formData.append(file.name, file);

    var response = this.httpClient.post<any>('https://xxuj660nkd.execute-api.ap-south-1.amazonaws.com/dev/upload', formData);
    return response;
  }
}
