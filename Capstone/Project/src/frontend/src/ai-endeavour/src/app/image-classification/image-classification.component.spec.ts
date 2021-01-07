import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ImageClassificationComponent } from './image-classification.component';

describe('ImageClassificationComponent', () => {
  let component: ImageClassificationComponent;
  let fixture: ComponentFixture<ImageClassificationComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ImageClassificationComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ImageClassificationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
