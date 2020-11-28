import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { ImagecaptionComponent } from './imagecaption/imagecaption.component';
import { TranslationComponent } from './translation/translation.component';
import { SpeechtotextComponent } from './speechtotext/speechtotext.component';

@NgModule({
   imports: [
      RouterModule.forRoot([
         { path: '', redirectTo: '/imagecaptionComponent', pathMatch: 'full' },
         { path: 'translationComponent', component: TranslationComponent },
         { path: 'imagecaptionComponent', component: ImagecaptionComponent },
         { path: 'speechtotextComponent', component: SpeechtotextComponent }
      ])
   ],
   exports: [ RouterModule ]
})
export class AppRoutingModule {}