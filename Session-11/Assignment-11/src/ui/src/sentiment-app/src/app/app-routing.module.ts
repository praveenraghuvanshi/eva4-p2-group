import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { ImagecaptionComponent } from './imagecaption/imagecaption.component';
import { TranslationComponent } from './translation/translation.component';

@NgModule({
   imports: [
      RouterModule.forRoot([
         { path: '', redirectTo: '/imagecaptionComponent', pathMatch: 'full' },
         { path: 'translationComponent', component: TranslationComponent },
         { path: 'imagecaptionComponent', component: ImagecaptionComponent }
      ])
   ],
   exports: [ RouterModule ]
})
export class AppRoutingModule {}