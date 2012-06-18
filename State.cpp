#include "State.h"


State::State() : original(),
                 P(),
                 dx(0.0),
                 dy(0.0),
                 dz(0.0),
                 thetaX(0.0),
                 thetaY(0.0),
                 thetaZ(0.0) {}

State::State(::Protein *p) : original(),
                             P(),
                             dx(0.0),
                             dy(0.0),
                             dz(0.0),
                             thetaX(0.0),
                             thetaY(0.0),
                             thetaZ(0.0) {
  for (int i = 0; i < p->length; ++i) {
    P.push_back(Point3D(p->sequence[i].Ca->x, p->sequence[i].Ca->y, p->sequence[i].Ca->z));
  }
  original = P;
}

void State::read(const char *file_name) {
  FILE *f = fopen(file_name, "r");
  char buff[1024];

  printf("Ucitavam podatke iz %s...\n", file_name);
  while (fgets(buff, sizeof(buff), f)) {
    double x, y, z;
    if (sscanf(buff, "ATOM %*d CA %*s %*s %*d %lf %lf %lf", &x, &y, &z)==3) {
      P.push_back(Point3D(x,y,z));
    }
  }
  original = P;
}

void State::transformMyself(double maxTranslacija, double maxRotacija) {
  dx = maxTranslacija - fmod(rand()*1.0, 2*maxTranslacija);
  dy = maxTranslacija - fmod(rand()*1.0, 2*maxTranslacija);
  dz = maxTranslacija - fmod(rand()*1.0, 2*maxTranslacija);
  thetaX = maxRotacija - 2*maxRotacija*rand()/RAND_MAX;
  thetaY = maxRotacija - 2*maxRotacija*rand()/RAND_MAX;
  thetaZ = maxRotacija - 2*maxRotacija*rand()/RAND_MAX;
  RotationMatrix rotacija = createRotationMatrix(thetaX, thetaY, thetaZ);
  Point3D pomak = getTranslation(dx, dy, dz);

  P.clear();
  for (int i = 0; i < (int)original.size(); ++i) {
    Point3D pomaknuta_tocka = rotacija * original[i] + pomak;
    P.push_back(pomaknuta_tocka);
  }
}

void State::transformOther(const State &A) {
  dx = A.dx + 5 - rand() % 11;
  dy = A.dy + 5 - rand() % 11;
  dz = A.dz + 5 - rand() % 11;
  thetaX = A.thetaX + 0.1 - 0.2*rand()/RAND_MAX;
  thetaY = A.thetaY + 0.1 - 0.2*rand()/RAND_MAX;
  thetaZ = A.thetaZ + 0.1 - 0.2*rand()/RAND_MAX;
  RotationMatrix rotacija = createRotationMatrix(thetaX, thetaY, thetaZ);
  Point3D pomak = getTranslation(dx, dy, dz);

  original = A.original;
  P.clear();
  for (int i = 0; i < (int)A.original.size(); ++i) {
    Point3D pomaknuta_tocka = rotacija*A.original[i] + pomak;
    P.push_back(pomaknuta_tocka);
  }
}

void State::reset() {
  dx = 0;
  dy = 0;
  dz = 0;
  thetaX = 0;
  thetaY = 0;
  thetaZ = 0;
  RotationMatrix rotacija = createRotationMatrix(thetaX, thetaY, thetaZ);
  Point3D pomak = getTranslation(dx, dy, dz);
  P.clear();
  for (int i = 0; i < (int)original.size(); ++i) {
    Point3D pomaknuta_tocka = rotacija * original[i] + pomak;
    P.push_back(pomaknuta_tocka);
  }
}
