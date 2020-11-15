import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Injectable } from '@angular/core';
@Injectable({
   providedIn: 'root'
})

export class ApiService {
  constructor(private httpClient: HttpClient) {}
  
  public translate(germanSentence: string): Observable<any> {
    console.log('Before translation: ' + germanSentence);
    var translationData = {
       germanText : germanSentence
    }
    var response = this.httpClient.post<any>('https://75t4f99u03.execute-api.ap-south-1.amazonaws.com/dev/translate',
    JSON.stringify(translationData));
    return response;
 }
}
