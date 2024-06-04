import mpi.*; // Импорт библиотеки MPI
import java.util.Arrays; // Импорт java.util.Arrays

public class Main { // Объявление класса Main

    static final int N = 100; // Объявление константы N = 100
    static final int MASTER = 0; // Объявление константы MASTER = 0

    public static void MatrixMultBlocking(String[] args){ // Объявление метода MatrixMultBlocking
        MPI.Init(args); // Инициализация MPI

        int currentProcess = MPI.COMM_WORLD.Rank(); // Получение номера текущего процесса
        int processesCount = MPI.COMM_WORLD.Size(); // Получение количества процессов

        if (N % processesCount != 0) { // Проверка на четность заданий для процессов
            if (currentProcess == MASTER) { // Если процесс - главный
                System.out.println("Odd number of tasks for"
                        + processesCount + " processes"); // Вывод сообщения
            }
        
            MPI.Finalize(); // Завершение работы MPI
            return; // Возврат из метода
        }
        int rowsPerProcess = N / processesCount; // Расчет количества строк на процесс

        Matrix B = new Matrix(N, N); // Создание матрицы B размером N x N
        Matrix A = new Matrix(N, N); // Создание матрицы A размером N x N
        Matrix C = new Matrix(N, N); // Создание матрицы C размером N x N

        double[] matrixASubFlatten = new double[rowsPerProcess * N]; // Создание массива для подматрицы A
        double[] matrixBFlatten = new double[N * N]; // Создание массива для матрицы B
        long startTime = 0; // Инициализация времени начала

       if (currentProcess == MASTER) { // Если процесс - главный
        startTime = System.currentTimeMillis(); // Инициализация времени начала

        A.fillMatrixWithNum(5); // Заполнение матрицы A числом 5
        B.fillMatrixWithNum(5); // Заполнение матрицы B числом 5

//            System.out.println("A: ");
//            A.print();
//
//            System.out.println("B: ");
//            B.print();

            matrixBFlatten = B.flatten(); // "Распрямление" матрицы B

            double[] matrixAFlatten = A.flatten(); // "Распрямление" матрицы A

            MPI.COMM_WORLD.Bcast(matrixBFlatten, 0, N * N, MPI.DOUBLE, MASTER); // Рассылка матрицы B по процессам

            MPI.COMM_WORLD.Scatter(matrixAFlatten, 0, rowsPerProcess * N, MPI.DOUBLE,  // Рассылаем данные из matrixAFlatten по процессам.
                    matrixASubFlatten, 0, rowsPerProcess * N, MPI.DOUBLE, MASTER);  // Собираем данные в matrixASubFlatten на MASTER процессе.

        }
        else {
            MPI.COMM_WORLD.Bcast(matrixBFlatten, 0, N * N, MPI.DOUBLE, MASTER);  // Транслируем данные из matrixBFlatten всем процессам.
            
            MPI.COMM_WORLD.Scatter(new double[N*N], 0, rowsPerProcess * N, MPI.DOUBLE,  // Рассылаем новую матрицу по процессам.
                    matrixASubFlatten, 0, rowsPerProcess * N, MPI.DOUBLE, MASTER);  // Собираем данные в matrixASubFlatten на MASTER процессе.
            B = new Matrix(matrixBFlatten, N, N);  // Инициализация объекта матрицы B.
        }

        Matrix subMatrixA = new Matrix(matrixASubFlatten, rowsPerProcess, N);  // Создаем подматрицу subMatrixA.
        Matrix resultMatrix = new Matrix(rowsPerProcess, N);  // Создаем матрицу для результата умножения.

        for (int i = 0; i < rowsPerProcess; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    resultMatrix.addValue(i, j, subMatrixA.getValue(i, k) * B.getValue(k, j));  // Вычисляем результирующую матрицу.
                }
            }
        }

        double[] resultMatrixFlatten = resultMatrix.flatten();  // Сглаживаем результат в одномерный массив.
        double[] matrixCFlatten = new double[N * N];  // Инициализация массива для матрицы C.

        MPI.COMM_WORLD.Gather(resultMatrixFlatten, 0, rowsPerProcess * N, MPI.DOUBLE,  // Собираем результаты умножения матрицы.
                matrixCFlatten, 0, rowsPerProcess * N, MPI.DOUBLE, MASTER);  // Собираем данные на MASTER процессе.

        if (currentProcess == MASTER) {
            C.setMatrixWithRowsOffset(matrixCFlatten, 0);  // Устанавливаем матрицу C на основе собранных данных.
            long endTime = System.currentTimeMillis();  // Фиксируем время окончания работы.
            System.out.println("Execution time of blocking collective: " + (endTime - startTime) + " ms for " + processesCount + " workers");  // Выводим время выполнения блокирующей коллективной операции.
//            System.out.println("Res: ");  // Выводим результат.
//            C.print();  // Печатаем матрицу C.
        }

        MPI.Finalize();  // Завершаем работу MPI.
    }

    public static void main(String[] args) {
        MatrixMultBlocking(args);  // Вызываем метод для умножения матриц.
    }
}