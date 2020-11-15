import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { AppComponent } from './app.component';
import { FirstComponent } from './first/first.component';
import { SecondComponent } from './second/second.component';
import { MenuComponent } from './menu/menu.component';
import { AppRoutingModule } from './app-routing.module';
import { HttpClientModule } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { TransferHttpCacheModule } from '@nguniversal/common';
import { NgtUniversalModule } from '@ng-toolkit/universal';
@NgModule({
   declarations: [
      AppComponent,
      FirstComponent,
      SecondComponent,
      MenuComponent
   ],
   imports: [
      BrowserModule.withServerTransition({ appId: 'serverApp' }),
      AppRoutingModule,
      HttpClientModule,
      CommonModule,
      TransferHttpCacheModule,
      NgtUniversalModule
   ],
   providers: [],
   bootstrap: [AppComponent]
})
export class AppModule { }