import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { SentimentAnalysisComponent } from './sentiment-analysis/sentiment-analysis.component';

@NgModule({
    imports: [
      RouterModule.forRoot([
        {
          path : '',
          redirectTo : 'sentimentAnalysisComponent',
          pathMatch: 'full'
        },
        {
          path : 'sentimentAnalysisComponent',
          component : SentimentAnalysisComponent
        }
      ])
    ],
    exports: [ RouterModule ]
})
export class AppRoutingModule { }
