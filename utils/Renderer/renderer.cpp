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
 *      p (distance of the camera from mass center of the model: r)
 *      c (color of background in RGB [0 - 255]: red, green, blue)
 *      n (square root of number of images: enter 3 to generate 9 images)
 *      f (path to PLY model)
 *      t (path of the folder where images will be generated)
 * ex: Renderer -g [-p r] [-c r g b] [-n Number] -f Path_to_model_file -t Path_to_image_folder
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

typedef double* Color;

using namespace std;

struct Cartesian {
    double x, y, z;
    Cartesian(double x, double y, double z): x(x), y(y), z(z) {}
    Cartesian(){x = 0; y = 0; z = 0;};
};

bool single_mode = false, group_mode = false;
Cartesian position;
Color background_color;
char *output_file_name = NULL, *model_file_name = NULL;
double camera_distance = 2.4;
int number_of_images = 1;

Cartesian from_polar(double r, double phi, double theta) {
    double x, y, z;
    x = r * cos(theta) * cos(phi);
    y = r * cos(theta) * sin(phi);
    z = r * sin(theta);
    return Cartesian(x, y, z);
}

void exit_with_help(){
    printf("3D Mesh (PLY) rendering\n"
                   "Usage 1 - single image:\n"
                   "Parameters:\n"
                   "      s (single mode)\n"
                   "      p (position of the camera in polar coordinates: r, phi[0 - 359], theta[0 - 359])\n"
                   "      c (color of background in RGB [0 - 255]: red, green, blue)\n"
                   "      f (path to PLY mesh to be rendered)\n"
                   "      t (PNG file name of generated image)\n"
                   " ex: Renderer -s [-p r phi theta] [-c r g b] -f Path_to_model_file -t Path_to_image_file\n\n"
                   " Usage 2 - group rendering:\n"
                   " Parameters:\n"
                   "      g (group mode)\n"
                   "      p (distance of the camera from mass center of the model: r)\n"
                   "      c (color of background in RGB [0 - 255]: red, green, blue)\n"
                   "      n (square root of number of images: enter 3 to generate 9 images)\n"
                   "      f (path to PLY model)\n"
                   "      t (path of the folder where images will be generated)\n"
                   " ex: Renderer -g [-p r] [-c r g b] [-n Number] -f Path_to_model_file -t Path_to_image_folder);\n");
    exit(1);
}

void parse_command_line(int argc, char **argv){
    int i = 1;
    while(i < argc){
        if (argv[i][0] != '-')
            break;
        switch(argv[i][1]){
            case 'h':
                exit_with_help();
                break;
            case 's':
                single_mode = true;
                break;
            case 'p':
                if (single_mode){
                    int r = (double)atoi(argv[++i]);
                    int phi = atoi(argv[++i]);
                    int theta = atoi(argv[++i]);
                    position =from_polar((double)r,(double)phi,(double)theta);
                }else if (group_mode){
                    camera_distance = (double)atoi(argv[++i]);
                    position = from_polar(camera_distance,0,0);
                }else{
                    exit_with_help();
                }
                break;
            case 'c':
                background_color[0] = (double)atoi(argv[++i]);
                background_color[0] = (double)atoi(argv[++i]);
                background_color[0] = (double)atoi(argv[++i]);
                break;
            case 'n':
                number_of_images = atoi(argv[++i]);
                break;
            case 'f':
                model_file_name = argv[++i];
                break;
            case 't':
                output_file_name = argv[++i];
                break;
            case 'g':
                group_mode = true;
                break;
        }
        i++;
    }
    if (!single_mode && !group_mode)
        exit_with_help();
    if (model_file_name == NULL || output_file_name == NULL)
        exit_with_help();
}

int main(int argc, char **argv) {

    double min_angle = 0;
    double max_angle = 360;

    background_color = new double[3];
    background_color[0] = 0;
    background_color[1] = 0;
    background_color[2] = 0;

    position = from_polar(camera_distance,0,0);

    parse_command_line(argc,argv);

    //read the model
    vtkSmartPointer<vtkPLYReader> reader =
            vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName (model_file_name);

    // Visualize
    vtkSmartPointer<vtkPolyDataMapper> mapper =
            vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());

    vtkSmartPointer<vtkActor> actor =
            vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    double* bound = mapper->GetBounds();
    Cartesian center((bound[1] - bound[0]) / 2, (bound[3] - bound[2]) / 2, (bound[5] - bound[4]) / 2);

    string output(output_file_name);
    string input(model_file_name);

    if (group_mode)
        output = output + input.substr(input.find_last_of('/')+1,input.find_last_of('.') - input.find_last_of('/') - 1);

    double step = (max_angle - min_angle)/number_of_images,theta = 0.0, phi = 0.0;

    for(int i = 0; i < number_of_images; i++) {
        for(int j = 0; j < number_of_images;j++) {
            vtkSmartPointer<vtkCamera> camera =
                    vtkSmartPointer<vtkCamera>::New();
            camera->SetPosition(position.x + center.x, position.y + center.y, position.z + center.z);
            camera->SetFocalPoint(center.x, center.y, center.z);

            vtkSmartPointer<vtkRenderer> renderer =
                    vtkSmartPointer<vtkRenderer>::New();
            renderer->SetActiveCamera(camera);
            renderer->AddActor(actor);
            renderer->SetBackground(background_color); // Black background

            vtkSmartPointer<vtkRenderWindow> renderWindow =
                    vtkSmartPointer<vtkRenderWindow>::New();
            renderWindow->AddRenderer(renderer);

            renderWindow->Render();

            vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter =
                    vtkSmartPointer<vtkWindowToImageFilter>::New();
            windowToImageFilter->SetInput(renderWindow);
            windowToImageFilter->SetMagnification(
                    3); //set the resolution of the output image (3 times the current resolution of vtk render window)
            windowToImageFilter->SetInputBufferTypeToRGB();
            windowToImageFilter->ReadFrontBufferOff(); // read from the back buffer
            windowToImageFilter->Update();

            vtkSmartPointer<vtkPNGWriter> writer =
                    vtkSmartPointer<vtkPNGWriter>::New();


            string suffix = "";
            if (group_mode)
                suffix = '_'+to_string((int)theta) + '_' + to_string((int)phi)+".png";

            writer->SetFileName((output+suffix).c_str());
            writer->SetInputConnection(windowToImageFilter->GetOutputPort());
            writer->Write();

            theta = step * (i+1);
            phi = step * (j+1);
            position = from_polar(camera_distance, phi,theta);
        }
    }
    return EXIT_SUCCESS;
}