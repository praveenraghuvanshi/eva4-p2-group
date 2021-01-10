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
  selectedImageSrc = '';
  classifiedImageResult = '';
  baseDirectory = '';
  uploadedImagesCount = 0
  accuracy = 0;
  model = '';
  uploading = false;
  uploaded = false;
  training = false;
  trained = false;
  classifying = false;
  previewImages = []
  classes = []
  class_to_idx = {}

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
      var filePath = files[0].webkitRelativePath;
      if(this.validateImageFolder(filePath) === false){
        alert("We are sorry, currently multiple dpeth of directory is not supported, Please upload directory with in the format Image Folder -> Class Folder -> Class Image file");
        return;
      }

      var path = files[0].webkitRelativePath;
      var Folder = path.split("/");
      console.log('Selected Folder: ' + Folder[0]);
      this.baseDirectory = Folder[0];
      console.log('Base Directory: ' + this.baseDirectory);

      this.classes = this.getClasses(files);
      this.uploadedImagesCount = files.length;
      var previewCount = this.uploadedImagesCount <= 10  ? this.uploadedImagesCount : 10;

      // Preview sample images - 10
      for (let i = 0; i < previewCount; i++) {
        var reader = new FileReader();
        reader.onload = (event:any) => {
          this.previewImages.push(event.target.result);
        }
        reader.readAsDataURL(event.target.files[i]);
      }

      // Clear data before upload of all files
      this.apiService.clear(this.baseDirectory).subscribe(data =>
        {
          console.log(JSON.stringify(data));
          console.log("Data deleted");

          // Upload to server
          const MAX_FILE_UPLOAD = 1000;
          var noOfFilesToUpload = files.length;
          if(files.length > MAX_FILE_UPLOAD){
            noOfFilesToUpload = MAX_FILE_UPLOAD;
          }
          for(let index = 0; index < noOfFilesToUpload; index++) {
            this.apiService.upload(files[index]).subscribe(data =>
              {
                if(index == noOfFilesToUpload - 1) {
                  this.uploading = false;
                  this.uploaded = true;
                  console.log("Uploaded " + index + " files");
                }
                setTimeout(() => {
                  // console.log("FIle uploaded: " + files[index].name);
                }, 100);
              },
              error => { //Error callback
                console.error('Error caught in component\n' + JSON.stringify(error))
                this.uploading = false;
                alert("An error occured while processing the request, Please retry the operation!!!");
              });
          }
        },
        error => { //Error callback
          console.error('error caught in component\nError Details:' + JSON.stringify(error));
          this.training = false;
          this.trained = true;
          alert("An error occured while processing the request, Please retry the operation!!!");
        });
    }
  }

  validateImageFolder(imagePath){
    if(imagePath !== ''){
      var pathSplit = imagePath.split("/");
      if(pathSplit.length != 3){
        return false;
      }
      return true;
    }
  }

  getClasses(files){
    var classes = []
    for (let index = 0; index < files.length; index++){
      var filePath = files[index].webkitRelativePath;
      if(filePath !== ''){
        var pathSplit = filePath.split("/");
        if(pathSplit.length > 3){
          alert("We are sorry, currently multiple dpeth of directory is not supported, Please upload directory with class as subdirectory and images inside the class directory")
          return;
        }
        var classname = filePath.split("/")[1];
        if(classes.includes(classname) == false){
          classes.push(classname);
        }
      }
    }
    console.log(classes);
    return classes;
  }

  trainModel(){
    this.training = true;
    this.apiService.train_ic(this.baseDirectory).subscribe(data =>
      {
        console.log(JSON.stringify(data));
        this.training = false;
        this.trained = true;
        this.accuracy = Math.round(data.test_acc * 100) / 100;
        this.model = data.model;
        this.class_to_idx = data.class_to_idx
        console.log("Training completed, Accuracy: " + this.accuracy);
      },
      error => { //Error callback
        console.error('error caught in component\nError Details:' + JSON.stringify(error));
        this.training = false;
        this.trained = true;
        alert("An error occured while processing the request, Please retry the operation!!!");
      });
  }

  classify(){
    this.classifying = true;
    this.apiService.classify(this.selectedImageSrc, this.model).subscribe(data =>
      {
        console.log(JSON.stringify(data));
        this.classifying = false;
        Object.entries(this.class_to_idx).forEach(([key, value]) => {
          console.log(key, value);
          if(value === data.result){
            this.classifiedImageResult = key;
          }
        });
        console.log("Classification Result: " + this.classifiedImageResult);
      },
      error => { //Error callback
        console.error('error caught in component\nError Details:' + JSON.stringify(error));
        this.classifying = false;
        alert("An error occured while processing the request, Please retry the operation!!!");
      });
  }
}
