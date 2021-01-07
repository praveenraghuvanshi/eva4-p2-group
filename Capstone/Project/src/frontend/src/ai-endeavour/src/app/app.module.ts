import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppComponent } from './app.component';
import { CommonModule } from '@angular/common';
import { TransferHttpCacheModule } from '@nguniversal/common';
import { HttpClientModule } from '@angular/common/http';
import { NgtUniversalModule } from '@ng-toolkit/universal';
import { SentimentAnalysisComponent } from './sentiment-analysis/sentiment-analysis.component'
import { MenuComponent } from './menu/menu.component'
import { FormsModule } from '@angular/forms';
import { AppRoutingModule } from './app-routing.module';
import { WelcomeComponent } from './welcome/welcome.component';
import { ImageClassificationComponent } from './image-classification/image-classification.component';

@NgModule({
  declarations: [
    AppComponent,
    SentimentAnalysisComponent,
    MenuComponent,
    WelcomeComponent,
    ImageClassificationComponent
  ],
  imports: [
    BrowserModule.withServerTransition({ appId: 'serverApp' }),
    CommonModule,
    TransferHttpCacheModule,
    HttpClientModule,
    NgtUniversalModule,
    FormsModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
