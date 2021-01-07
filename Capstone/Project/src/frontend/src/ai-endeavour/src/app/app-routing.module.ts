import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { WelcomeComponent } from './welcome/welcome.component';
import { ImageClassificationComponent} from './image-classification/image-classification.component';
import { SentimentAnalysisComponent } from './sentiment-analysis/sentiment-analysis.component';


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
        },
        {
          path : 'ic',
          component : ImageClassificationComponent
        }
      ])
    ],
    exports: [ RouterModule ]
})
export class AppRoutingModule { }
