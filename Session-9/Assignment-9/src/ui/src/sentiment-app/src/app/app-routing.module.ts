import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { TranslationComponent } from './translation/translation.component';

@NgModule({
   imports: [
      RouterModule.forRoot([
         { path: '', redirectTo: '/translationComponent', pathMatch: 'full' },
         { path: 'translationComponent', component: TranslationComponent }
      ])
   ],
   exports: [ RouterModule ]
})
export class AppRoutingModule {}