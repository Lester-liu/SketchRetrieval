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

using namespace std;

typedef double* Color; // 3 double numbers
enum Mode {Single, Group};

struct Cartesian {
    double x, y, z;

    Cartesian(double x, double y, double z): x(x), y(y), z(z) {}

    Cartesian() {
        x = 0;
        y = 0;
        z = 0;
    };
};

Cartesian from_polar(double r, double phi, double theta) {
    double x, y, z;
    x = r * cos(theta) * cos(phi);
    y = r * cos(theta) * sin(phi);
    z = r * sin(theta);
    return Cartesian(x, y, z);
}

string output, input; // model and image file name

Mode mode = Single;
Color background; // model background color

Cartesian position; // camera position
double camera_distance = 2.4; // initial distance form the mass center of the object
double phi, theta; // camera angles

int step_count = 1;
double min_angle = 0;
double max_angle = 360;

void show_help(){
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
}

/*
 * Process all arguments
 */
bool parse_command_line(int argc, char **argv) {

    int i = 1;
    while(i < argc) {
        if (argv[i][0] != '-')
            break;
        switch(argv[i][1]) {
            case 'h': // help
                show_help();
                return false;
            case 's': // single mode flag
                mode = Single;
                break;
            case 'g': // group mode
                mode = Group;
                break;
            case 'p': // position
                switch (mode) {
                    case Single:
                        camera_distance = (double)atoi(argv[++i]);
                        phi = (double)atoi(argv[++i]);
                        theta = (double)atoi(argv[++i]);
                        position = from_polar(camera_distance, phi, theta);
                        break;
                    case Group:
                        camera_distance = (double)atoi(argv[++i]);
                        position = from_polar(camera_distance, 0, 0);
                        break;
                    default:
                        show_help();
                        return false;
                }
                break;
            case 'c': // color
                background[0] = (double)atoi(argv[++i]) / 255.0;
                background[1] = (double)atoi(argv[++i]) / 255.0;
                background[2] = (double)atoi(argv[++i]) / 255.0;
                break;
            case 'n': // square root of number of images
                step_count = atoi(argv[++i]);
                break;
            case 'f': // input model file
                input = argv[++i];
                break;
            case 't': // output image file
                output = argv[++i];
                break;
        }
        i++;
    }
    if (input.length() == 0 || output.length() == 0) { // invalid file name
        show_help();
        return false;
    }
    return true;
}

int main(int argc, char **argv) {

    background = new double[3];
    background[0] = 0;
    background[1] = 0;
    background[2] = 0;

    position = from_polar(camera_distance, 0, 0); // initial position

    if (!parse_command_line(argc, argv))
        return EXIT_FAILURE;

    // Read the model
    vtkSmartPointer<vtkPLYReader> reader = vtkSmartPointer<vtkPLYReader>::New();
    reader->SetFileName(input.c_str());

    // Visualize
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(reader->GetOutputPort());

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);

    double* bound = mapper->GetBounds();
    Cartesian center((bound[1] - bound[0]) / 2, (bound[3] - bound[2]) / 2, (bound[5] - bound[4]) / 2);

    // common prefix
    if (mode == Group)
        output = output + input.substr(input.find_last_of('/') + 1, input.find_last_of('.') - input.find_last_of('/') - 1);

    // compute delta angle of each iteration
    double step = (max_angle - min_angle) / step_count;

    // use all possible angle combinations
    for(int i = 0; i < step_count; i++, theta += step) {
        for(int j = 0; j < step_count; j++, phi += step) {

            position = from_polar(camera_distance, phi,theta);

            vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();
            camera->SetPosition(position.x + center.x, position.y + center.y, position.z + center.z);
            camera->SetFocalPoint(center.x, center.y, center.z);

            vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
            renderer->SetActiveCamera(camera);
            renderer->AddActor(actor);
            renderer->SetBackground(background); // Black background

            vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
            renderWindow->AddRenderer(renderer);

            renderWindow->Render();

            vtkSmartPointer<vtkWindowToImageFilter> windowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
            windowToImageFilter->SetInput(renderWindow);
            windowToImageFilter->SetMagnification(2); //set the resolution of the output image
            windowToImageFilter->SetInputBufferTypeToRGB();
            windowToImageFilter->ReadFrontBufferOff(); // read from the back buffer
            windowToImageFilter->Update();

            vtkSmartPointer<vtkPNGWriter> writer =
                    vtkSmartPointer<vtkPNGWriter>::New();

            // use angles in the file name
            string suffix = "";
            if (mode == Group)
                suffix = '_' + to_string((int)theta) + '_' + to_string((int)phi) + ".png";

            writer->SetFileName((output + suffix).c_str());
            writer->SetInputConnection(windowToImageFilter->GetOutputPort());
            writer->Write();

        }
    }

    delete[] background;

    return EXIT_SUCCESS;
}