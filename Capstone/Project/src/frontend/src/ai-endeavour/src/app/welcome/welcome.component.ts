import { Component, OnInit } from '@angular/core';
declare var $: any;

@Component({
  selector: 'app-welcome',
  templateUrl: './welcome.component.html',
  styleUrls: ['./welcome.component.css']
})
export class WelcomeComponent implements OnInit {

  constructor() { }

  ngOnInit() {
    $(document).ready(function(){
      $(".carousel").carousel(
        {
          interval:2000
        });
      console.log("Inside Carousel");
    });
  }

}
