__kernel void convolution2D(
    __global int * inputData, __global int * outputData, __constant int * maskData,
    int width, int height, int maskWidth,  int imageChannels, int stride){
    //@@ Insert code to implement matrix multiplication here
    //printf("Starting kernel code..\n");
    int x = get_global_id(0);
    int y = get_global_id(1);
    int maskRadius = maskWidth / 2;
    //if (maskWidth != 1) {maskRadius = maskWidth / 2;} // this is integer division

    //printf("maskRadius: %d\n", maskRadius);
    int outputWidth = (width - maskWidth ) / stride + 1;
    int outputHeight = (height - maskWidth ) / stride + 1;
    //int accum = 0;
    //printf("Starting for loop...");
    //for (int i = 0 - maskRadius; i < maskRadius; i++)
    if (x < outputHeight && y < outputWidth)
    {
        for (int k = 0; k < imageChannels; k++)
        {
            int accum = 0;
            for (int j = -maskRadius; j <= maskRadius; j++)
            {
                for (int i = -maskRadius; i <= maskRadius; i++)
                {
                int kx = x * stride + maskRadius;
                int ky = y * stride + maskRadius;
                int xOffset = ky + i;
                int yOffset = kx + j;
                if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height)
                    {
                        int imagePixel = inputData[(yOffset*width + xOffset)*imageChannels + k];
                        // printf("imagePixel: %d\n", imagePixel);
                        int maskValue = maskData[(j+maskRadius)*maskWidth + i + maskRadius];
                        // printf("maskValue: %d\n", maskValue);
                        accum += imagePixel * maskValue;
                        // accum += inputData[((y + j)*width + (x+i))*imageChannels + k];
                    }
                     
                }
            }
            //pixels are in the range of 0 to 1
            // printf("accum: %d\n", accum);
            outputData[(y * outputWidth + x)*imageChannels + k] = accum;
            // printf("outputData[%d]: %d\n", (x * width + y)*imageChannels + k, 
            //                     outputData[(x * width + y)*imageChannels + k]);
        }
    }
}
