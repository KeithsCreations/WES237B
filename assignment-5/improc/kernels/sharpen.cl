__constant float gaussian_kernel[3][3] = {
    {0.0947416f, 0.118318f, 0.0947416f},
    {0.118318f, 0.147761f, 0.118318f},
    {0.0947416f, 0.118318f, 0.0947416f}
};

__constant float sharpening_kernel[3][3] = {
    {0.0f, -1.0f, 0.0f},
    {-1.0f, 5.0f, -1.0f},
    {0.0f, -1.0f, 0.0f}
};

__constant float h_edge_kernel[3][3] = {
    {-1.0f, -1.0f, -1.0f},
    {0.0f, 0.0f, 0.0f},
    {1.0f, 1.0f, 1.0f}
};

__constant float v_edge_kernel[3][3] = {
    {-1.0f, 0.0f, 1.0f},
    {-1.0f, 0.0f, 1.0f},
    {-1.0f, 0.0f, 1.0f}
};

__kernel void imgproc(__global const int* input, __global uint* output, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    // float effect[3][3];
    
    // if (select == 0) { effect = gaussian_kernel; }
    // else if (select == 1) { effect = sharpening_kernel; }
    // else if (select == 2) { effect = h_edge_kernel; }
    // else if (select == 3) { effect = v_edge_kernel; }

    if (x < width && y < height) {
        float accum = 0.0f;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pixel_x = clamp(x + i, 0, width - 1);
                int pixel_y = clamp(y + j, 0, height - 1);
                accum += input[pixel_y * width + pixel_x] * sharpening_kernel[i + 1][j + 1];
            }
        }
        
        // printf("Value at index %d: %f\n", y * width + x, accum);
        output[y * width + x] = accum;
    }
}
    
