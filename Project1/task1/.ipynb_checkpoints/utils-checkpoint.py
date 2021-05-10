{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "\n",
    "def calculate_rms(img1, img2):\n",
    "    \"\"\"\n",
    "    Calculates RMS error between two images. Two images should have same sizes.\n",
    "    \"\"\"\n",
    "    if (img1.shape[0] != img2.shape[0]) or \\\n",
    "            (img1.shape[1] != img2.shape[1]) or \\\n",
    "            (img1.shape[2] != img2.shape[2]):\n",
    "        raise Exception(\"img1 and img2 should have sime sizes.\")\n",
    "\n",
    "    diff = np.abs(img1.astype(np.int) - img2.astype(np.int))\n",
    "    return np.sqrt(np.mean(diff ** 2))\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
