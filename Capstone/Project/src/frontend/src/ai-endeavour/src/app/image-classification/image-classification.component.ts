import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-image-classification',
  templateUrl: './image-classification.component.html',
  styleUrls: ['./image-classification.component.css']
})
export class ImageClassificationComponent {
  projectname = ''
  images = []

  constructor() { }

  OnSelectFolder(event){
    if (event.target.files && event.target.files[0]){
      var files = event.target.files;
      var path = files[0].webkitRelativePath;
      var Folder = path.split("/");
      console.log('Selected Folder: ' + Folder[0]);
      var filesCount = files.length;
      var previewCount = filesCount <= 10  ? filesCount : 10;
      for (let i = 0; i < previewCount; i++) {
              var reader = new FileReader();
              reader.onload = (event:any) => {
                this.images.push(event.target.result);
              }
              reader.readAsDataURL(event.target.files[i]);
      }
    }
  }
}
