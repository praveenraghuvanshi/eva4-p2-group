import { Component } from '@angular/core';


@Component({
   selector: 'app-menu',
   styleUrls: ['./menu.component.css'],
   template: `
   <div class="header">
      <ul>
         <li><a href="/">Home</a></li>
         <li><a href="ic">Image Classification</a></li>
         <li><a href="sa">Sentiment Analysis</a></li>
      </ul>
</div>
   `,
   styles: [`
      :host {margin: 0; padding: 0}
      ul {list-style-type: none; padding: 0;}
      li {display: inline-block;}

   `]
})
export class MenuComponent {}
