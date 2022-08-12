#ifndef NNVIEWER_H
#define NNVIEWER_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class NNViewer; }
QT_END_NAMESPACE

class NNViewer : public QMainWindow
{
    Q_OBJECT

public:
    NNViewer(QWidget *parent = nullptr);
    ~NNViewer();

private:
    Ui::NNViewer *ui;

    void populateMainMenu();
    void openNNConfigFile();
};
#endif // NNVIEWER_H
