import { BrowserModule, BrowserTransferStateModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { AppComponent } from './app.component';
import { FirstComponent } from './first/first.component';
import { SecondComponent } from './second/second.component';
import { MenuComponent } from './menu/menu.component';
import { AppRoutingModule } from './app-routing.module';
import { HttpClientModule } from '@angular/common/http';
import { AppModule } from './app.module';
@NgModule({
   imports: [
      
      AppRoutingModule,
      HttpClientModule,
      AppModule,
      BrowserTransferStateModule
   ],
   providers: [],
   bootstrap: [AppComponent]
})
export class AppBrowserModule { }