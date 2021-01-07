import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { SentimentAnalysisComponent } from './sentiment-analysis/sentiment-analysis.component';
import { WelcomeComponent } from './welcome/welcome.component';

@NgModule({
    imports: [
      RouterModule.forRoot([
        {
          path : '',
          component : WelcomeComponent
        },
        {
          path : 'sa',
          component : SentimentAnalysisComponent
        }
      ])
    ],
    exports: [ RouterModule ]
})
export class AppRoutingModule { }
