
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <windows.h>

using namespace std;
using namespace cv;
using namespace cv::detail;


// Nastavenie parametrov pre Stitcher class
vector<String> img_names;
bool try_cuda = false;
double work_megapix = 0.6;                         // Parameter udavajuci o rozliseni pri registracii obrazu
double seam_megapix = 0.1;                         // Parameter udavajuci o rozliseni, ktore je potrebne pri procese odhadu trhlin
double compose_megapix = -1;
float conf_thresh = 0.4f;                           // Prahova hodnota pre dva obrazky z panoramy
string features_type = "surf";
float match_conf = 0.65f;
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
string warp_type = "spherical";                     // Typ zakrivenia obrazu
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;    // Metoda kompenzacie expozicie
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
string seam_find_type = "gc_color";             // Metoda odhadu trhlin v obraze
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;
string result_name = "stitched_img";
int range_width = -1;



static int parseFolderImages(int offset, int num_of_pictures)
{

    vector<String> fn;
    glob("C:/Users/kacma/Desktop/bc_panorama/screenshots/*.png", fn, false);
    size_t count = fn.size();           // Pocet png suborov v adresari
    if (count == 0) {
        return 1;
    }
    if ((num_of_pictures + offset) > fn.size()) {       // Kontrola konca nacitavania suborov zo zlozky
        return -1;
    }
    for (size_t i = offset; i < (num_of_pictures + offset); i++) {
        img_names.push_back(fn[i]);
        cout << "Proceeding to stitch: ";
        cout << fn[i];
        cout << "\n";
    }

    return 0;
}


int main(int argc, char* argv[])
{

#if 0
    cv::setBreakOnError(true);
#endif
    string path = "D:/Programs/UnityHub/panorama/Assets/Resources/";
    int counter = 0;
    int offset = 0;
    while (1) {
        int retval = parseFolderImages(offset, 2);

        if (retval == -1) {
            return -1;

        }
        else if (retval) {
            Sleep(1235);
            continue;
        }
        // Kontrola dostatku obrazkov
        int num_images = static_cast<int>(img_names.size());
        if (num_images < 2)
        {
            cout << "Nedostatok obrazkov potrebnych na spojenie obrazu.";
            img_names.clear();
            counter++;
            offset += 2;
            continue;
        }

        double work_scale = 1, seam_scale = 1, compose_scale = 1;
        bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

        Ptr<Feature2D> finder;
        finder = xfeatures2d::SURF::create();

        Mat full_img, img;
        vector<ImageFeatures> features(num_images);
        vector<Mat> images(num_images);
        vector<Size> full_img_sizes(num_images);
        double seam_work_aspect = 1;

        for (int i = 0; i < num_images; ++i)
        {
            full_img = imread(samples::findFile(img_names[i]));
            full_img_sizes[i] = full_img.size();

            if (full_img.empty())
            {
                cout << "Nenaslo obrazok";
                img_names.clear();
                counter++;
                continue;
            }
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
            if (!is_seam_scale_set)
            {
                seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
                seam_work_aspect = seam_scale / work_scale;
                is_seam_scale_set = true;
            }

            computeImageFeatures(finder, img, features[i]);
            features[i].img_idx = i;

            resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
            images[i] = img.clone();
        }

        full_img.release();
        img.release();

        vector<MatchesInfo> pairwise_matches;
        Ptr<FeaturesMatcher> matcher;
        matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);

        (*matcher)(features, pairwise_matches);
        matcher->collectGarbage();


        // Ponechanie obrazkov z rovnakej panoramy
        vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
        vector<Mat> img_subset;
        vector<String> img_names_subset;
        vector<Size> full_img_sizes_subset;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            img_names_subset.push_back(img_names[indices[i]]);
            img_subset.push_back(images[indices[i]]);
            full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
        }

        images = img_subset;
        img_names = img_names_subset;
        full_img_sizes = full_img_sizes_subset;

        // Kontrola dostatocneho mnozstva obrazkov
        num_images = static_cast<int>(img_names.size());
        if (num_images < 2)
        {
            cout << "Nedostatocny pocet obrazkov\n";
            img_names.clear();
            counter++;
            offset += 2;
            continue;
        }
        Ptr<Estimator> estimator;
        estimator = makePtr<HomographyBasedEstimator>();

        vector<CameraParams> cameras;
        if (!(*estimator)(features, pairwise_matches, cameras))
        {
            cout << "Odhad matice homografie zlyhal\n";
            Mat replacement = imread(samples::findFile(img_names[0]));
            imwrite(path + result_name + "_" + to_string(counter) + ".png", replacement);
            img_names.clear();
            counter++;
            offset += 2;
            continue;
        }

        for (size_t i = 0; i < cameras.size(); ++i)
        {
            Mat R;
            cameras[i].R.convertTo(R, CV_32F);
            cameras[i].R = R;
        }

        Ptr<detail::BundleAdjusterBase> adjuster;
        adjuster = makePtr<detail::BundleAdjusterRay>();
        adjuster->setConfThresh(conf_thresh);
        Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
        if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
        if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
        if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
        if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
        if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
        adjuster->setRefinementMask(refine_mask);
        if (!(*adjuster)(features, pairwise_matches, cameras))          // Nenajdenie dostatocneho poctu klucovych bodov
        {
            cout << "\nNenasiel sa dostatocny pocet klucovych bodov, pridavam nahradny obrazok \n\n";
            Mat replacement = imread(samples::findFile(img_names[0]));
            imwrite(path + result_name + "_" + to_string(counter) + ".png", replacement);
            counter++;
            offset += 2;
            continue;
        }

        // Najdenie medianu ohniskovej vzdialenosti
        vector<double> focals;
        for (size_t i = 0; i < cameras.size(); ++i)
        {
            focals.push_back(cameras[i].focal);
        }

        sort(focals.begin(), focals.end());
        float warped_image_scale;
        if (focals.size() % 2 == 1)
            warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
        else
            warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];

        vector<Point> corners(num_images);
        vector<UMat> masks_warped(num_images);
        vector<UMat> images_warped(num_images);
        vector<Size> sizes(num_images);
        vector<UMat> masks(num_images);

        // Pripravenie masiek obrazkov
        for (int i = 0; i < num_images; ++i)
        {
            masks[i].create(images[i].size(), CV_8U);
            masks[i].setTo(Scalar::all(255));
        }

        // Ohybanie obrazkov a ich masiek
        Ptr<WarperCreator> warper_creator;
        warper_creator = makePtr<cv::SphericalWarper>();

        Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

        for (int i = 0; i < num_images; ++i)
        {
            Mat_<float> K;
            cameras[i].K().convertTo(K, CV_32F);
            float swa = (float)seam_work_aspect;
            K(0, 0) *= swa; K(0, 2) *= swa;
            K(1, 1) *= swa; K(1, 2) *= swa;

            corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
            sizes[i] = images_warped[i].size();

            warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
        }

        vector<UMat> images_warped_f(num_images);
        for (int i = 0; i < num_images; ++i)
            images_warped[i].convertTo(images_warped_f[i], CV_32F);

        Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
        if (dynamic_cast<GainCompensator*>(compensator.get()))
        {
            GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
            gcompensator->setNrFeeds(expos_comp_nr_feeds);
        }

        if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
        {
            ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
            ccompensator->setNrFeeds(expos_comp_nr_feeds);
        }

        if (dynamic_cast<BlocksCompensator*>(compensator.get()))
        {
            BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
            bcompensator->setNrFeeds(expos_comp_nr_feeds);
            bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
            bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
        }

        compensator->feed(corners, images_warped, masks_warped);

        Ptr<SeamFinder> seam_finder;
        seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);

        seam_finder->find(images_warped_f, corners, masks_warped);

        // Uvolnenie nepouzitej pamate
        images.clear();
        images_warped.clear();
        images_warped_f.clear();
        masks.clear();

        Mat img_warped, img_warped_s;
        Mat dilated_mask, seam_mask, mask, mask_warped;
        Ptr<Blender> blender;
        double compose_work_aspect = 1;

        for (int img_idx = 0; img_idx < num_images; ++img_idx)
        {

            // Precitanie a v pripade potreby zmenenie rozmeru obrazku
            full_img = imread(samples::findFile(img_names[img_idx]));
            if (!is_compose_scale_set)
            {
                if (compose_megapix > 0)
                    compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
                is_compose_scale_set = true;

                // Vypocet relativnych mier
                compose_work_aspect = compose_scale / work_scale;

                // Aktualizovanie miery ohnuteho obrazku
                warped_image_scale *= static_cast<float>(compose_work_aspect);
                warper = warper_creator->create(warped_image_scale);

                // Aktualizovanie rozmerov a rohov
                for (int i = 0; i < num_images; ++i)
                {
                    cameras[i].focal *= compose_work_aspect;
                    cameras[i].ppx *= compose_work_aspect;
                    cameras[i].ppy *= compose_work_aspect;

                    Size sz = full_img_sizes[i];
                    if (std::abs(compose_scale - 1) > 1e-1)
                    {
                        sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                        sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                    }

                    Mat K;
                    cameras[i].K().convertTo(K, CV_32F);
                    Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                    corners[i] = roi.tl();
                    sizes[i] = roi.size();
                }
            }
            if (abs(compose_scale - 1) > 1e-1)
                resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
            else
                img = full_img;
            full_img.release();
            Size img_size = img.size();

            Mat K;
            cameras[img_idx].K().convertTo(K, CV_32F);

            // Ohnutie obrazku
            warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

            // Ohnutie masky obrazku
            mask.create(img_size, CV_8U);
            mask.setTo(Scalar::all(255));
            warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

            // Kompenzacia expozicie
            compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

            img_warped.convertTo(img_warped_s, CV_16S);
            img_warped.release();
            img.release();
            mask.release();

            dilate(masks_warped[img_idx], dilated_mask, Mat());
            resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
            mask_warped = seam_mask & mask_warped;

            if (!blender)
            {
                blender = Blender::createDefault(blend_type, try_cuda);
                Size dst_sz = resultRoi(corners, sizes).size();
                float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
                if (blend_width < 1.f)
                    blender = Blender::createDefault(Blender::NO, try_cuda);
                else if (blend_type == Blender::MULTI_BAND)
                {
                    MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                    mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
                }

                blender->prepare(corners, sizes);
            }

            // Zahladenie sucasneho obrazku
            blender->feed(img_warped_s, mask_warped, corners[img_idx]);
        }

        Mat result, result_mask;
        blender->blend(result, result_mask);

        resize(result, result, Size(640, 480));      // Zmena velkosti na povodnu 640x480
        imwrite(path + result_name + "_" + to_string(counter) + ".png", result);

        counter++;
        offset += 2;
        cout << "\n";
        img_names.clear();
    }
    return 0;
}
