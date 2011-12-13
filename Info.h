#pragma once
class Info
{
 public:
	Info(void);
  Info(int, int);
	~Info(void);

  int diagonal() const { return diagonal_; }
  int offset() const { return offset_; }

 private:
  int diagonal_, offset_;
};

