import { BrowserModule, BrowserTransferStateModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { AppModule } from './app.module';

@NgModule({
  imports: [
    
  AppModule,
    
  BrowserTransferStateModule],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppBrowserModule { }
