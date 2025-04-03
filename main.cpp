#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace std;
using namespace Eigen;

Vector2d palu(MatrixXd A, Vector2d b)
{
	FullPivLU<MatrixXd> lu_decomp(A);
	Vector2d x = lu_decomp.solve(b);
	
	return x;
}

Vector2d qr(MatrixXd A, Vector2d b)
{
	FullPivHouseholderQR <MatrixXd> qr(A);
	Vector2d x = qr.solve(b);
	
	return x;
}
	

int main()
{
	Vector2d Xex(-1.0e+0,-1.0e+00);
	
	MatrixXd A1 {{5.547001962252291e-01,-3.770900990025203e-02},
	{8.320502943378437e-01,-9.992887623566787e-01}};
	Vector2d b1(-5.169911863249772e-01, 1.672384680188350e-01);
	
	MatrixXd A2 {{5.547001962252291e-01,-5.540607316466765e-01},
	{8.320502943378437e-01,-8.324762492991313e-01}};
	Vector2d b2(-6.394645785530173e-04, 4.259549612877223e-04);

	MatrixXd A3 {{5.547001962252291e-01,-5.547001955851905e-01},
	{8.320502943378437e-01,-8.320502947645361e-01}};
	Vector2d b3(-6.400391328043042e-10, 4.266924591433963e-10);
	
	// Soluzioni con decomposizoni PALU
	Vector2d x1PALU = palu(A1,b1);
	Vector2d x2PALU = palu(A2,b2);
	Vector2d x3PALU = palu(A3,b3);
	
	// Soluzioni con decomposizoni PALU
	Vector2d x1QR = qr(A1,b1);
	Vector2d x2QR = qr(A2,b2);
	Vector2d x3QR = qr(A3,b3);
	
	// Errori relativi
	
	double err1PALU = (Xex-x1PALU).norm()/Xex.norm();
	double err2PALU = (Xex-x2PALU).norm()/Xex.norm();
	double err3PALU = (Xex-x3PALU).norm()/Xex.norm();
	
	double err1QR = (Xex-x1QR).norm()/Xex.norm();
	double err2QR = (Xex-x2QR).norm()/Xex.norm();
	double err3QR = (Xex-x3QR).norm()/Xex.norm();
	
	cout << std::scientific << setprecision(2);
	cout << " " <<endl;
	// Stampa PALU
	cout << "Using PALU, first solution X1 : "<< "[ " << x1PALU[0] <<" , "<< x1PALU[1] <<" ]"<< endl;
	cout << "Relative error : "<<err1PALU << endl;
	cout << " " <<endl;
	cout << "Using PALU, second solution X2 : "<< "[ " << x2PALU[0] <<" , "<< x2PALU[1] <<" ]"<< endl;
	cout << "Relative error : "<<err2PALU << endl;
	cout << " " <<endl;
	cout << "Using PALU, third solution X3 : "<< "[ " << x3PALU[0] <<" , "<< x3PALU[1] <<" ]"<< endl;
	cout << "Relative error : "<<err3PALU << endl;
	cout << " " <<endl;
	
	//Stampa QR
	cout << "Using QR, first solution X1 : "<< "[ " << x1QR[0] <<" , " <<x1QR[1] <<" ]"<< endl;
	cout << "Relative error : "<<err1QR << endl;
	cout << " " <<endl;
	cout << "Using QR, second solution X2 : "<< "[ " << x2QR[0] <<" , "<< x2QR[1] <<" ]"<< endl;
	cout << "Relative error : "<<err2QR << endl;
	cout << " " <<endl;
	cout << "Using QR, third solution X3 : "<< "[ " << x3QR[0] <<" , "<< x3QR[1] <<" ]"<< endl;
	cout << "Relative error : "<<err3QR << endl;
	cout << " " <<endl;
	return 0;
}