/*
 * 3D Mesh (PLY) rendering
 *
 * In the project, we will show how to draw 3D mesh with VTK library, you can
 * use this code to generate snapshot of your model from any point of view
 * you like.
 *
 * Usage 1 - single image:
 * Parameters:
 *      s (single mode)
 *      p (position of the camera in polar coordinates: r, phi[0 - 359], theta[0 - 359])
 *      c (color of background in RGB [0 - 255]: red, green, blue)
 *      f (path to PLY mesh to be rendered)
 *      t (PNG file name of generated image)
 * ex: Renderer -s [-p r theta phi] [-c r g b] -f Path_to_model_file -t Path_to_image_file
 *
 * Usage 2 - group rendering:
 * Parameters:
 *      g (group mode)
 *      p (distance of the camera from mass center of the model: d)
 *      c (color of background in RGB [0 - 255]: red, green, blue)
 *      n (square root of number of images: enter 3 to generate 9 images)
 *      f (path to PLY model)
 *      t (path of the folder where images will be generated)
 * ex: Renderer -g [-p d] [-c r g b] [-n Number] -f Path_to_model_file -t Path_to_image_folder
 */

#include <vtkPolyData.h>
#include <vtkPLYReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCamera.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

#include <cmath>

using namespace std;

struct Cartesian {
    double x, y, z;
    Cartesian(double x, double y, double z): x(x), y(y), z(z) {}
};

Cartesian from_polar(double r, double phi, double theta) {
    double x, y, z;
    x = r * cos(theta) * cos(phi);
    y = r * cos(theta) * sin(phi);
    z = r * sin(theta);
    return Cartesian(x, y, z);
}

int main(int argc, char *argv[]) {
    string mesh;
    double step = 10; // Number of poses
    double min_angle = 0;
    double max_angle = 360;

    if(argc != 2)
        mesh = "/home/lyx/workspace/cuda/Sketch/pipeline/models_ply/m0.ply";
    else
        mesh = argv[1];

    vtkSmartPointer<vtkPLYReader> reader =
            vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName (mesh.c_str());

    // Visualize
    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());

    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    double* bound = mapper->GetBounds();
    Cartesian center((bound[1] - bound[0]) / 2, (bound[3] - bound[2]) / 2, (bound[5] - bound[4]) / 2);
    
    vtkSmartPointer<vtkCamera> camera =
            vtkSmartPointer<vtkCamera>::New();
    Cartesian coord = from_polar(2.4, 90, 0);
    camera->SetPosition(coord.x + center.x, coord.y + center.y, coord.z + center.z);
    camera->SetFocalPoint(center.x, center.y, center.z);

    vtkSmartPointer<vtkRenderer> renderer =
            vtkSmartPointer<vtkRenderer>::New();
    renderer->SetActiveCamera(camera);
    renderer->AddActor(actor);
    renderer->SetBackground(0, 0, 0); // Black background

    vtkSmartPointer<vtkRenderWindow> renderWindow =
            vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    renderWindow->Render();

    vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter =
            vtkSmartPointer<vtkWindowToImageFilter>::New();
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetMagnification(3); //set the resolution of the output image (3 times the current resolution of vtk render window)
    windowToImageFilter->SetInputBufferTypeToRGB();
    windowToImageFilter->ReadFrontBufferOff(); // read from the back buffer
    windowToImageFilter->Update();

    vtkSmartPointer<vtkPNGWriter> writer =
            vtkSmartPointer<vtkPNGWriter>::New();
    writer->SetFileName("/home/lyx/workspace/cuda/Sketch/pipeline/views/m0_0_0.png");
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    writer->Write();

    return EXIT_SUCCESS;
}