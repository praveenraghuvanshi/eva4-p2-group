import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { FirstComponent } from './first/first.component';
import { SecondComponent } from './second/second.component';
import { TranslationComponent } from './translation/translation.component';

@NgModule({
   imports: [
      RouterModule.forRoot([
         { path: '', redirectTo: '/translationComponent', pathMatch: 'full' },
         { path: 'translationComponent', component: TranslationComponent },
         { path: 'firstComponent', component: FirstComponent },
         { path: 'secondComponent', component: SecondComponent }
      ])
   ],
   exports: [ RouterModule ]
})
export class AppRoutingModule {}