import { Component, OnInit } from '@angular/core';
import { Observable } from 'rxjs';
import { ApiService } from '../api.service';

@Component({
  selector: 'app-image-classification',
  templateUrl: './image-classification.component.html',
  styleUrls: ['./image-classification.component.css']
})
export class ImageClassificationComponent {
  public response: Observable<any>;
  projectname = ''
  selectedImageSrc = '';
  classifiedImage = '';
  uploadedImagesCount = 0
  uploading = false;
  uploaded = false;
  training = false;
  trained = false;
  predicting = false;
  previewImages = []
  classes = []

  constructor(private apiService: ApiService) { }

  onSelectFile(event) { // called each time file input changes
    if (event.target.files && event.target.files[0]){
      this.uploading = true;
      const [image] = event.target.files;
      var reader = new FileReader();
      reader.readAsDataURL(image);
      reader.onload = () => {
        this.selectedImageSrc = reader.result as string;
      }
    }
  }

  OnSelectFolder(event){
    if (event.target.files && event.target.files[0]){
      this.uploading = true;
      var files = event.target.files;
      // this.classes = this.extractClasses(files);
      for (let index = 0; index < files.length; index++){
        var filePath = files[index].webkitRelativePath;
        if(filePath !== ''){
          var pathSplit = filePath.split("/");
          if(pathSplit.length > 3){
            alert("We are sorry, currently multiple dpeth of directory is not supported, Please upload directory with class as subdirectory and images inside the class directory")
            return;
          }
          var classname = filePath.split("/")[1];
          if(this.classes.includes(classname) == false){
            console.log(classname);
            this.classes.push(classname);
          }
        }
      }
      console.log(this.classes);
      var path = files[0].webkitRelativePath;
      var Folder = path.split("/");
      console.log('Selected Folder: ' + Folder[0]);
      var filesCount = files.length;
      this.uploadedImagesCount = filesCount;
      var previewCount = filesCount <= 10  ? filesCount : 10;
      for (let i = 0; i < previewCount; i++) {
              var reader = new FileReader();
              reader.onload = (event:any) => {
                this.previewImages.push(event.target.result);
                if(i == 9){
                  this.uploaded = true;
                  this.uploading = false;

                  // Upload to server
                  for(let index = 0; index < files.length; index++){
                    this.apiService.upload(files[index]).subscribe(data =>
                      {
                        console.log('Response: ' + JSON.stringify(data));
                        if(index == files.length - 1){
                          this.uploading = false;
                        }
                      },
                      error => { //Error callback
                        console.error('Error caught in component\n' + JSON.stringify(error))
                        this.uploading = false;
                        alert("An error occured while processing the request, Please retry the operation!!!");
                      });
                  }
                }
              }
              reader.readAsDataURL(event.target.files[i]);
      }
    }
  }

  trainModel(){
    this.training = true;
    setTimeout(() => {
      this.trained = true;
      this.training = false;
    }, 5000);
  }

  classify(){
    this.predicting = true;
    setTimeout(() => {
      this.predicting = false;
      this.classifiedImage = "Cat";
    }, 5000);
  }
}
